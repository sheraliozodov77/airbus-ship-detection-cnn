# app.py
import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
import os

# -----------------------------
# Utility Functions
# -----------------------------

def preprocess_image(image, target_size=(384, 384)):
    """
    Resize the image to the specified target size and normalize pixel values to [0, 1].
    """
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype("float32") / 255.0
    return image_normalized

def predict_mask(image, model, target_size=(384, 384), threshold=0.5):
    """
    Preprocess the image, run model inference, and return a binary mask.
    
    Parameters:
    - image: Input image array (RGB).
    - model: The pretrained segmentation model.
    - target_size: Tuple (width, height) for model input.
    - threshold: Threshold to binarize the model's probability output.
    
    Returns:
    - binary_mask: Predicted binary mask of shape (target_size[1], target_size[0], 1).
    """
    processed = preprocess_image(image, target_size)
    input_tensor = np.expand_dims(processed, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]  # shape: (384,384,1)
    binary_mask = (pred > threshold).astype(np.uint8)
    return binary_mask

def mask_to_rle(mask):
    """
    Convert a binary mask to run-length encoding (RLE).
    
    Parameters:
    - mask: A 2D binary mask.
    
    Returns:
    - rle: A string of space-separated values representing the RLE.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(str(x) for x in runs)
    return rle

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay the binary mask on the original image.
    
    If the mask size differs from the image size, it will be resized.
    The mask is overlayed in red.
    
    Parameters:
    - image: Original image (RGB).
    - mask: Binary mask (2D or 3D).
    - alpha: Transparency factor for overlay.
    
    Returns:
    - overlay: Image with mask overlay.
    """
    # Resize mask if necessary
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Convert 2D mask to 3 channels if required
    if mask.ndim == 2:
        mask = np.stack([mask] * 3, axis=-1)
    # Create a red overlay
    color_mask = np.zeros_like(image)
    color_mask[..., 0] = 255  # Red channel
    overlay = cv2.addWeighted(image, 1.0, color_mask * mask, alpha, 0)
    return overlay

# -----------------------------
# Model Loading with New Caching API
# -----------------------------
@st.cache_resource
def load_trained_model(model_choice):
    """
    Load and cache the trained model based on the selected model type.
    Model files should be stored in the "models/" directory.
    """
    if model_choice == "ResUNet":
        model_path = os.path.join("models", "resunet_best.keras")
    elif model_choice == "U-Net++":
        model_path = os.path.join("models", "unetpp_best.keras")
    elif model_choice == "Attention U-Net":
        model_path = os.path.join("models", "att_unet_best.keras")
    else:
        st.error("Invalid Model Choice")
        return None
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(
    page_title="Airbus Ship Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar: Model Selection & Threshold Setting
# -----------------------------
st.sidebar.title("Model & Threshold Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model for Ship Detection:",
    ("ResUNet", "U-Net++", "Attention U-Net")
)

# Single threshold slider for inference (default: 0.5)
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("### Upload an image for inference")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the selected model
model = load_trained_model(model_choice)

st.title("Airbus Ship Detection")
st.write("Upload an image to see the predicted ship segmentation overlayed on the original image.")

# -----------------------------
# Main App Logic: Inference and Display
# -----------------------------
if uploaded_file is not None:
    # Read the uploaded file using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image and prediction side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        try:
            st.image(image, use_container_width=True)
        except TypeError:
            st.image(image, use_column_width=True)
    
    # Run inference on the uploaded image
    start_time = time.time()
    pred_mask = predict_mask(image, model, target_size=(384, 384), threshold=threshold)
    exec_time = time.time() - start_time

    # Create an overlay visualization
    overlay = overlay_mask(image, pred_mask, alpha=0.5)
    
    with col2:
        st.subheader("Predicted Mask Overlay")
        try:
            st.image(overlay, use_container_width=True)
        except TypeError:
            st.image(overlay, use_column_width=True)
    
    st.write(f"Inference Time: {exec_time:.2f} seconds")
    
    # Check if any ship is detected and convert mask to RLE if so.
    if np.sum(pred_mask) == 0:
        st.info("No ship detected in the uploaded image.")
        rle_string = ""
    else:
        rle_string = mask_to_rle(pred_mask.squeeze())
        st.subheader("Run-Length Encoding")
        st.text(rle_string)
    
    st.download_button("Download RLE as Text", data=rle_string, file_name="prediction_rle.txt")
else:
    st.info("Please upload an image to see predictions.")

st.markdown("---")
st.markdown("Developed by [Sherali Ozodov](https://github.com/sheraliozodov77)")
