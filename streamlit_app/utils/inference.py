# utils/inference.py
import tensorflow as tf
import os
from utils.preprocess import preprocess_input_image

def load_trained_model(model_choice):
    """
    Load the pretrained model based on the choice.
    Assumes model files are stored in the 'models/' directory.
    """
    model_path = ""
    if model_choice == "ResUNet":
        model_path = os.path.join("models", "resunet_best.keras")
    elif model_choice == "U-Net++":
        model_path = os.path.join("models", "unetpp_best.keras")
    elif model_choice == "Attention U-Net":
        model_path = os.path.join("models", "att_unet_best.keras")
    else:
        raise ValueError("Invalid model choice")
    # Load model without compiling (for inference)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict_mask(image, model, image_size=(384, 384)):
    """
    Preprocess the input image, run inference, and return a binary mask.
    """
    # Preprocess: resize and normalize the image
    processed_image = preprocess_input_image(image, image_size)
    input_tensor = tf.expand_dims(processed_image, axis=0)
    pred = model.predict(input_tensor, verbose=0)[0]
    # Threshold prediction to create binary mask
    binary_mask = (pred > 0.5).astype("uint8")
    return binary_mask