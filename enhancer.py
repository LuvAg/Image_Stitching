import numpy as np
from PIL import Image
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from mirnet_custom import MIRNetBlock  # custom layer registration
from huggingface_hub import from_pretrained_keras


@st.cache_resource
def load_enhancer_model():
    model_path = os.path.join("models", "mirnet_enhancer.h5")
    return load_model(model_path, custom_objects={
        "TFOpLambda": lambda x: x,  # or adjust if specific op is needed
        "MIRNetBlock": MIRNetBlock  # ensure this is correctly imported
    })




model = load_enhancer_model()


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def enhance_image(image_path):
    input_image = load_and_preprocess_image(image_path)
    enhanced_image = model.predict(input_image)
    enhanced_image = np.squeeze(enhanced_image) * 255.0
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

def enhance_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    saved_files = []
    for file in image_files:
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        print(f"✨ Enhancing: {file}")
        enhanced_img = enhance_image(input_path)
        enhanced_img.save(output_path)
        saved_files.append(output_path)
    return saved_files

def enhance_image_pil(pil_img):
    pil_img = pil_img.resize((256, 256))
    img_array = np.array(pil_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    enhanced_image = model.predict(img_array)
    enhanced_image = np.squeeze(enhanced_image) * 255.0
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)
