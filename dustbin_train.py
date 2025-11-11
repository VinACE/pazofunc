# -*- coding: utf-8 -*-
"""
Dustbin Detection YOLOv11 Training Script (Local GPU Version)
Cleaned and adapted for /opt/ working directory

Author: Devi Nanda & ChatGPT
"""

import os
from glob import glob
import yaml
import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow

# ==========================================================
# 🧱 CONFIG
# ==========================================================
HOME = "/opt"
DATA_DIR = f"{HOME}/datasets/data/dustbin"
YAML_PATH = f"{DATA_DIR}/data.yaml"

# Roboflow defaults - prefer setting ROBofLOW_API_KEY env var
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "Cr5DRDHLGniU7LctIWeV")
ROBOFLOW_WORKSPACE = "dustbindetection"
ROBOFLOW_PROJECT = "dustbin_detection-5802a"
ROBOFLOW_VERSION = 4

# ==========================================================
# 📦 STEP 1: Install & Download Data (if run in fresh env)
# ==========================================================
os.system("pip install -q ultralytics roboflow torchvision pyyaml matplotlib pillow opencv-python")

print("✅ Libraries installed (or already present).")

# ==========================================================
# 🧩 STEP 2: Download dataset from Roboflow
# ==========================================================
dataset_location = None
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    # download in YOLO format (compatible with ultralytics)
    dataset = version.download("yolov8")
    dataset_location = dataset.location
    print("📂 Dataset downloaded to:", dataset_location)
    print("Files:", os.listdir(dataset_location))
except Exception as e:
    print("⚠️ Roboflow download failed:", e)
    print("Make sure ROBOFLOW_API_KEY, workspace, project and version are correct.")

# ==========================================================
# 🗂️ STEP 3: Copy Dataset to /opt/datasets/data/dustbin
# ==========================================================
def make_ds(root, path_to_copy):
    os.makedirs(path_to_copy, exist_ok=True)
    files = glob(f"{root}/*")
    for file in files:
        name = os.path.basename(file)
        dest = os.path.join(path_to_copy, name)
        if os.path.isdir(file):
            # copy recursively
            os.system(f"cp -r '{file}' '{dest}'")
        elif os.path.isfile(file):
            os.system(f"cp '{file}' '{dest}'")

if dataset_location:
    make_ds(root=dataset_location, path_to_copy=DATA_DIR)
    print("✅ Dataset copied to:", DATA_DIR)
    os.system(f"ls -ltr {DATA_DIR}")
else:
    print("⚠️ No dataset location available; skipping copy step.")

# ==========================================================
# 🧾 STEP 4: Fix data.yaml paths (absolute paths)
# ==========================================================
def fix_yaml_paths(original_yaml_path, final_yaml_path, data_dir):
    if not os.path.exists(original_yaml_path):
        print(f"⚠️ Original data.yaml not found at {original_yaml_path}")
        return False

    with open(original_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Update train/val/test to absolute paths using data_dir
    # Roboflow usually stores images under "train/images", "valid/images", or "val/images" — try common keys
    # Find candidates inside original data to map to our layout
    def choose_path(key_names):
        for k in key_names:
            if k in data:
                return data[k]
        return None

    train_rel = choose_path(["train", "train/images", "train/images/train"])
    val_rel = choose_path(["val", "valid", "valid/images", "val/images"])
    test_rel = choose_path(["test", "test/images"])

    # If Roboflow used 'valid' instead of 'val'
    if train_rel is None:
        # try to infer by presence of folders
        if os.path.exists(os.path.join(data_dir, "train")):
            train_rel = os.path.join(data_dir, "train")

    # Build absolute paths pointing into our DATA_DIR layout
    data["train"] = os.path.join(data_dir, "train")
    data["val"] = os.path.join(data_dir, "valid") if os.path.exists(os.path.join(data_dir, "valid")) else os.path.join(data_dir, "val")
    data["test"] = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else data.get("test", data["val"]) 

    # Write fixed yaml
    os.makedirs(os.path.dirname(final_yaml_path), exist_ok=True)
    with open(final_yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"✅ Fixed YAML written to: {final_yaml_path}")
    return True

# Try to locate original data.yaml inside the downloaded dataset folder
orig_yaml = None
if dataset_location:
    candidates = [
        os.path.join(dataset_location, "data.yaml"),
        os.path.join(dataset_location, "dataset.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            orig_yaml = c
            break

if orig_yaml:
    success = fix_yaml_paths(orig_yaml, YAML_PATH, DATA_DIR)
else:
    print("⚠️ Could not find original data.yaml in the downloaded dataset. You may need to supply a data.yaml at", YAML_PATH)

# ==========================================================
# 🎨 STEP 5: Visualization Helper (inspect images & bboxes)
# ==========================================================
class Visualization:
    def __init__(self, data_types, n_ims, rows, cmap="rgb"):
        self.data_types = data_types
        self.n_ims = n_ims
        self.rows = rows
        self.cmap = cmap
        self.colors = ["firebrick", "darkorange", "blueviolet"]
        self.class_dict = {}
        if os.path.exists(YAML_PATH):
            self.get_cls_names()
        self.get_bboxes()

    def get_cls_names(self):
        with open(YAML_PATH, "r") as file:
            data = yaml.safe_load(file)
        # data['names'] might be list/dict
        names = data.get("names")
        if isinstance(names, dict):
            # keys may be strings; convert to int-indexed dict
            self.class_dict = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            self.class_dict = {i: n for i, n in enumerate(names)}
        else:
            self.class_dict = {}

    def get_bboxes(self):
        self.vis_datas, self.analysis_datas, self.im_paths = {}, {}, {}
        for data_type in self.data_types:
            all_bboxes, all_analysis_datas = [], {}
            im_dir_candidates = [
                os.path.join(DATA_DIR, data_type, "images"),
                os.path.join(DATA_DIR, data_type),
                os.path.join(DATA_DIR, data_type, "img"),
            ]
            im_paths = []
            for c in im_dir_candidates:
                if os.path.exists(c):
                    im_paths = glob(f"{c}/*")
                    break
            for im_path in im_paths:
                lbl_path = im_path
                # label path mapping: replace images with labels and .jpg/.png -> .txt
                if "/images/" in im_path:
                    lbl_path = im_path.replace("/images/", "/labels/")
                else:
                    lbl_path = im_path.replace(".jpg", ".txt").replace(".png", ".txt")
                lbl_path = lbl_path.rsplit(".", 1)[0] + ".txt"

                if not os.path.isfile(lbl_path):
                    continue
                with open(lbl_path) as f:
                    lines = f.readlines()
                bboxes = []
                for data in lines:
                    parts = data.strip().split()[:5]
                    try:
                        cls_name = self.class_dict.get(int(parts[0]), str(parts[0]))
                    except Exception:
                        cls_name = parts[0]
                    bboxes.append([cls_name] + [float(x) for x in parts[1:]])
                    all_analysis_datas[cls_name] = all_analysis_datas.get(cls_name, 0) + 1
                all_bboxes.append(bboxes)
            self.vis_datas[data_type] = all_bboxes
            self.analysis_datas[data_type] = all_analysis_datas
            self.im_paths[data_type] = im_paths

    def plot(self, rows, cols, count, im_path, bboxes):
        plt.subplot(rows, cols, count)
        or_im = np.array(Image.open(im_path).convert("RGB"))
        height, width, _ = or_im.shape
        for bbox in bboxes:
            class_id, x_center, y_center, w, h = bbox
            x_min = int((x_center - w / 2) * width)
            y_min = int((y_center - h / 2) * height)
            x_max = int((x_center + w / 2) * width)
            y_max = int((y_center + h / 2) * height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(or_im, (x_min, y_min), (x_max, y_max), color, 2)
        plt.imshow(or_im)
        plt.axis("off")
        plt.title(f"{len(bboxes)} object(s)")
        return count + 1

    def vis(self, save_name):
        if save_name not in self.vis_datas or len(self.vis_datas[save_name]) == 0:
            print(f"⚠️ No visualizable images for {save_name}")
            return
        plt.figure(figsize=(25, 20))
        cols = max(1, self.n_ims // self.rows)
        indices = random.sample(range(len(self.vis_datas[save_name])), min(self.n_ims, len(self.vis_datas[save_name])))
        count = 1
        for i in indices:
            im_path, bboxes = self.im_paths[save_name][i], self.vis_datas[save_name][i]
            count = self.plot(self.rows, cols, count, im_path, bboxes)
        plt.show()

    def data_analysis(self, save_name, color):
        cls_names = list(self.analysis_datas.get(save_name, {}).keys())
        counts = list(self.analysis_datas.get(save_name, {}).values())
        if not cls_names:
            print(f"⚠️ No class distribution data for {save_name}")
            return
        _, ax = plt.subplots(figsize=(30, 10))
        ax.bar(cls_names, counts, color=color)
        ax.set_title(f"{save_name.upper()} Class Distribution")
        plt.xticks(rotation=90)
        plt.show()

    def run(self):
        for i, name in enumerate(self.data_types):
            self.data_analysis(name, self.colors[i % len(self.colors)])
            self.vis(name)

# ==========================================================
# 📊 STEP 6: Run Visualization (best-effort)
# ==========================================================
vis = Visualization(data_types=["train", "val", "test", "valid"], n_ims=12, rows=4, cmap="rgb")
vis.run()

# ==========================================================
# 🚀 STEP 7: Train YOLOv11
# ==========================================================
# Choose base model: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
MODEL_NAME = "yolo11n.pt"

model = YOLO(MODEL_NAME)

train_args = {
    "data": YAML_PATH if os.path.exists(YAML_PATH) else (orig_yaml if orig_yaml else None),
    "epochs": 50,
    "imgsz": 640,
    "device": 0,  # GPU; set to 0 or 'cpu' if needed
    "project": "/opt/runs/dustbin",  # Custom output dir
    "name": "yolo11_dustbin",       # Experiment name
    "save": True,
    "save_period": 1,
}

if train_args["data"] is None:
    raise SystemExit("No data.yaml found. Cannot start training. Please provide a valid dataset/data.yaml.")

print("🚀 Starting training with args:", {k: train_args[k] for k in ["data", "epochs", "imgsz", "project", "name"]})

train_results = model.train(**train_args)

print(f"✅ Training complete. Results saved in {os.path.join(train_args['project'], train_args['name'])}")

# ==========================================================
# 💾 STEP 8: Save Best Model to a Known Location
# ==========================================================
best_model_path = os.path.join(train_args["project"], train_args["name"], "weights", "best.pt")
final_save_path = "/opt/models/dustbin_yolo11_best.pt"

os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

if os.path.exists(best_model_path):
    os.system(f"cp '{best_model_path}' '{final_save_path}'")
    print(f"✅ Best model saved at: {final_save_path}")
else:
    print("⚠️ Could not find best.pt — training may not have completed successfully.")

# ==========================================================
# 📈 STEP 9: Optional Evaluation & Quick Inference
# ==========================================================
try:
    print("\n📈 Evaluating model on validation set...")
    metrics = model.val()
    print(metrics)
except Exception as e:
    print("⚠️ Evaluation failed:", e)

# Quick inference test (optional) — save predictions to runs/detect/predict
test_image_dir = os.path.join(DATA_DIR, "test", "images") if os.path.exists(os.path.join(DATA_DIR, "test", "images")) else None
if test_image_dir and os.path.exists(test_image_dir):
    test_imgs = [p for p in os.listdir(test_image_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(test_imgs) > 0:
        img_path = os.path.join(test_image_dir, test_imgs[0])
        try:
            results = model.predict(img_path, save=True, imgsz=640, conf=0.4)
            print("✅ Prediction done. Check 'runs/detect/predict' for results.")
        except Exception as e:
            print("⚠️ Quick inference failed:", e)
    else:
        print("⚠️ No test images found for prediction.")
else:
    print("⚠️ Test image directory not found for quick inference.")
