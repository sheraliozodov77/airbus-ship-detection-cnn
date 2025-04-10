# utils/preprocess.py
import cv2

def preprocess_input_image(image, target_size=(384, 384)):
    """
    Resize an image to the target size and normalize pixel values to [0,1].
    
    Parameters:
        image (ndarray): The input image in BGR or RGB format.
        target_size (tuple): The desired image size (width, height).
        
    Returns:
        image (ndarray): Resized and normalized image.
    """
    # Resize using OpenCV
    image_resized = cv2.resize(image, target_size)
    # Convert pixel values to float32 in range [0, 1]
    image_normalized = image_resized.astype("float32") / 255.0
    return image_normalized