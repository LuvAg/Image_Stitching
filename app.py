import os
import streamlit as st
from PIL import Image
from io import BytesIO
import shutil
import subprocess

st.set_page_config(page_title="SuperGlue Panorama Stitching App", layout="centered")
st.title("📸 SuperGlue Panorama Stitching App")

st.markdown("### Upload multiple images to stitch into a panorama")

# Allow uploading multiple image files
uploaded_files = st.file_uploader("Upload image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Create or clear the temp image folder
    input_dir = "uploaded_images"
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)

    # Save the uploaded images to the folder
    for file in uploaded_files:
        image = Image.open(file)
        image_path = os.path.join(input_dir, file.name)
        image.save(image_path)

    st.success(f"Uploaded {len(uploaded_files)} images to `{input_dir}`")

    # Run the stitching pipeline (adjust path or command as needed)
    with st.spinner("Running stitching process..."):
        result = subprocess.run(["python", "match_pairs.py", "--input_dir", input_dir], capture_output=True, text=True)
        st.text(result.stdout)

    # Display the final stitched image if exists
    final_image_path = sorted(os.listdir("."), reverse=True)
    final_image_path = next((f for f in final_image_path if f.startswith("1_") and f.endswith(".jpg")), None)

    if final_image_path and os.path.exists(final_image_path):
        st.image(final_image_path, caption="🧵 Stitched Panorama", use_column_width=True)
    else:
        st.error("❌ Could not find the final stitched image.")
