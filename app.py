# app.py
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
        # Path handling
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        input_dir = os.path.abspath(os.path.join(BASE_DIR, "uploaded_images"))
        method_key = method.lower().replace(" ", "_")
        output_dir = os.path.abspath(os.path.join(BASE_DIR, input_dir, f"output_{method_key}"))
        
        # Clean and prepare input/output folders
        try:
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir)
            os.makedirs(input_dir, exist_ok=True, mode=0o755)
            
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True, mode=0o755)
            
        except Exception as dir_error:
            st.error(f"🚨 Directory setup failed: {dir_error}")
            st.stop()

        st.info("Saving images...")
        for idx, file in enumerate(uploaded_files, start=1):
            try:
                image = Image.open(file).convert("RGB")
                if enhance:
                    st.text(f"Enhancing image {idx}...")
                    image = enhance_image_pil(image)
                image.save(os.path.join(input_dir, f"{idx}.jpg"))
            except Exception as save_error:
                st.error(f"💾 Failed to save image {idx}: {save_error}")
                st.stop()

        # Generate temp_pairs.txt
        pairs_path = os.path.join(input_dir, "temp_pairs.txt")
        try:
            with open(pairs_path, "w") as f:
                for i in range(1, len(uploaded_files)):
                    f.write(f"{i}.jpg {i+1}.jpg\n")
        except IOError as file_error:
            st.error(f"📄 Failed to create pairs file: {file_error}")
            st.stop()

        start_idx = 1
        end_idx = len(uploaded_files)

        st.info(f"Running **{method}** on {end_idx} images...")
        st.session_state.stitch_success = False

        with st.spinner("🧠 Stitching in progress..."):
            try:
                if method == "Original Sequential":
                    result_img = original_stitch(start_idx, end_idx, input_dir, output_dir)
                elif method == "Hierarchical Stitching":
                    result_filename = hierarchical_stitching(start_idx, end_idx, input_dir, output_dir)
                    result_img = cv2.imread(os.path.join(output_dir, result_filename))
                elif method == "Continuous Stitching":
                    result_img = continuous_stitch(start_idx, end_idx, input_dir, output_dir)

                if result_img is not None:
                    stitched_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(Image.fromarray(stitched_rgb), caption="🧵 Final Stitched Panorama", use_column_width=True)
                    st.success("✅ Stitching complete!")
                    st.session_state.stitch_success = True
                else:
                    raise ValueError("Empty result image")

            except Exception as e:
                st.error(f"❌ Stitching failed: {str(e)}")
                st.session_state.stitch_success = False

        # Cleanup logic
        if "stitch_success" in st.session_state:
            try:
                if st.session_state.stitch_success:
                    shutil.rmtree(output_dir, ignore_errors=True)
                    # Preserve input dir but remove contents
                    for item in os.listdir(input_dir):
                        item_path = os.path.join(input_dir, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
            except Exception as cleanup_error:
                st.error(f"🧹 Cleanup failed: {cleanup_error}")

        # Reset uploader
        st.session_state.uploader_key = f"uploader_{int(time.time())}"