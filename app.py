import os
import shutil
import streamlit as st
from PIL import Image
import cv2
import time

# Stitching methods
from stitching import iterative_superglue_stitch as original_stitch
from hierarchical_stitch import hierarchical_stitching
from continuous_stitch import run_pipeline as continuous_stitch

# --- Streamlit App Config ---
st.set_page_config(page_title="SuperGlue Panorama Stitching", layout="centered")
st.title("📸 SuperGlue Panorama Stitching App")

# === Dynamic key to reset file_uploader ===
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "uploader_0"
# Modified enhancer import
from enhancer import enhance_image_pil


# === Sidebar: Options ===
st.sidebar.title("🧵 Stitching Options")
method = st.sidebar.selectbox(
    "Choose a stitching algorithm",
    ["Original Sequential", "Hierarchical Stitching", "Continuous Stitching"]
)

enhance = st.sidebar.checkbox("✨ Enhance images before stitching")

# === Upload Section ===
st.markdown("Upload your image sequence (we'll rename them to 1.jpg, 2.jpg, etc.):")

uploaded_files = st.file_uploader(
    "Upload at least two images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

# === Main Processing Block ===
if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("❗ Please upload at least two images.")
    else:
        input_dir = "uploaded_images"
        method_key = method.lower().replace(" ", "_")
        output_dir = os.path.join(input_dir, f"output_{method_key}")

        # Clean and prepare input/output folders
        if os.path.exists(input_dir):
            for filename in os.listdir(input_dir):
                file_path = os.path.join(input_dir, filename)
                if os.path.isfile(file_path) or filename.endswith(".npz") or filename == "temp_pairs.txt":
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(input_dir, exist_ok=True)

        os.makedirs(output_dir, exist_ok=True)

        st.info("Saving images...")
        for idx, file in enumerate(uploaded_files, start=1):
            image = Image.open(file).convert("RGB")

            if enhance:
                st.text(f"Enhancing image {idx}...")
                image = enhance_image_pil(image)

            image.save(os.path.join(input_dir, f"{idx}.jpg"))

        start_idx = 1
        end_idx = len(uploaded_files)

        st.info(f"Running **{method}** on {end_idx} images...")

        with st.spinner("🧠 Stitching in progress..."):
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

                if method == "Original Sequential":
                    result_img = original_stitch(start_idx, end_idx, input_dir, output_dir)

                elif method == "Hierarchical Stitching":
                    result_filename = hierarchical_stitching(start_idx, end_idx, input_dir, output_dir)
                    result_img = cv2.imread(os.path.join(output_dir, result_filename))

                elif method == "Continuous Stitching":
                    result_img = continuous_stitch(start_idx, end_idx, input_dir, output_dir)

                stitched_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(stitched_rgb), caption="🧵 Final Stitched Panorama", use_column_width=True)
                st.success("✅ Stitching complete!")

            except Exception as e:
                st.error(f"❌ Stitching failed: {e}")

            finally:
                # Cleanup output_dir
                shutil.rmtree(output_dir, ignore_errors=True)

                # Clean uploaded_images content
                for filename in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, filename)
                    if os.path.isfile(file_path) or filename.endswith(".npz") or filename == "temp_pairs.txt":
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                # Clean up any global temp files
                if os.path.exists("temp_pairs.txt"):
                    os.remove("temp_pairs.txt")

                # Reset uploader by changing its key and rerun
                st.session_state.uploader_key = f"uploader_{int(time.time())}"
                # No need for st.experimental_rerun(), the app will effectively "reset" now.
