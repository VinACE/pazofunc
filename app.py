import streamlit as st
import requests
import os
from datetime import datetime
import logging
from io import BytesIO 

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
AZURE_FUNCTION_KEY = os.getenv("AZURE_FUNCTION_KEY")
# Example:
AZURE_FUNCTION_URL = "https://cavin-pazzo-20251015-ci.azurewebsites.net/api/Upload_image?code="

BLOB_BASE_URL = "https://pazouploadetest.blob.core.windows.net/images"

# ---------------------------------------------------------------
# LOGGER SETUP
# ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="YOLOv11 Inference Portal", page_icon="🧠", layout="centered")

st.title("🧠 YOLOv11 Smart Detection Portal")
st.markdown("Upload images for **Dress Code** or **Dustbin** analysis — then submit for inference.")

# ---------------------------------------------------------------
# SECTION 1: DRESS CODE UPLOAD
# ---------------------------------------------------------------
st.header("👔 Section 1: Dress Code Detection")
dresscode_files = st.file_uploader(
    "Upload one or more images for Dress Code check",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dresscode_uploader",
)

if dresscode_files:
    st.write("Uploaded Dress Code Files:")
    for file in dresscode_files:
        st.markdown(f"- {file.name}")

# ---------------------------------------------------------------
# SECTION 2: DUSTBIN UPLOAD
# ---------------------------------------------------------------
st.header("🗑️ Section 2: Dustbin Detection")
dustbin_files = st.file_uploader(
    "Upload one or more images for Dustbin detection",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="dustbin_uploader",
)

if dustbin_files:
    st.write("Uploaded Dustbin Files:")
    for file in dustbin_files:
        st.markdown(f"- {file.name}")

# ---------------------------------------------------------------
# COMMON SUBMIT BUTTON
# ---------------------------------------------------------------
if st.button("🚀 Submit All"):
    if not dresscode_files and not dustbin_files:
        st.warning("Please upload at least one image in either section before submitting.")
    else:
        st.info("Uploading files and triggering inference... Please wait.")
        results = []

        # --- Helper: Upload to Azure Blob via Function App ---
        def upload_and_infer(file, category):
            try:
                filename_prefix = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"
                logger.info(f"Uploading {filename_prefix} to Azure Function...")

                files = {"file": (filename_prefix, file.getvalue(), file.type)}

                response = requests.post(AZURE_FUNCTION_URL, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    result["filename"] = filename_prefix
                    result["category"] = category
                    return result
                else:
                    logger.error(f"Inference failed: {response.text}")
                    return {
                        "filename": filename_prefix,
                        "category": category,
                        "status": "error",
                        "message": response.text,
                    }

            except Exception as e:
                logger.exception("Error during upload/inference")
                return {
                    "filename": file.name,
                    "category": category,
                    "status": "error",
                    "message": str(e),
                }

        # --- Process Dresscode Files ---
        for file in dresscode_files or []:
            result = upload_and_infer(file, "dresscode")
            results.append(result)

        # --- Process Dustbin Files ---
        for file in dustbin_files or []:
            result = upload_and_infer(file, "dustbin")
            results.append(result)

        st.success("✅ All images processed!")
        st.subheader("📊 Inference Results")

        for res in results:
            st.markdown(f"### 🖼️ {res.get('filename')}")
            if res.get("status") == "error":
                st.error(res.get("message"))
            else:
                st.json(res)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.caption("Powered by YOLOv11 • Azure Functions • Streamlit UI")
