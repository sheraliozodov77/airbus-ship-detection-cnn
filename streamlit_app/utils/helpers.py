# utils/helpers.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask_to_rle(mask):
    """
    Convert a binary mask to run-length encoding (RLE).
    Pixels are read in column-major order.
    
    Parameters:
        mask (ndarray): A binary mask (2D array) with values 0 and 1.
        
    Returns:
        rle (str): The run-length encoded string.
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(str(x) for x in runs)
    return rle

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay a binary mask onto an image.
    
    If mask dimensions differ from the image, the mask is resized to match.
    The overlay color is red.
    
    Parameters:
        image (ndarray): Original image (RGB), e.g. shape (H, W, 3)
        mask (ndarray): Binary mask, e.g. shape (h, w, 1) or (h, w)
        alpha (float): Transparency factor for the mask overlay.
        
    Returns:
        overlay (ndarray): The image with the mask overlayed.
    """
    # Resize mask if needed to match image dimensions.
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # If mask is 2D, convert it to 3 channels.
    if mask.ndim == 2:
        mask = np.stack([mask] * 3, axis=-1)
    
    # Create a red color mask.
    color_mask = np.zeros_like(image)
    color_mask[..., 0] = 255  # Red channel

    # Compute weighted overlay: Original image + alpha * (color_mask multiplied by the binary mask)
    overlay = cv2.addWeighted(image, 1.0, (color_mask * mask), alpha, 0)
    return overlay