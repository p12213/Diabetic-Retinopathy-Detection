"""
Diabetic Retinopathy Computer Vision Preprocessing Module
Implements Ben Graham's fundus cropping, Green Channel isolation, CLAHE, and vessel enhancement.
"""

import os
import cv2
import numpy as np
from PIL import Image

def crop_image_from_gray(img, tol=7):
    """
    Crops out black border background around fundus circular image (Ben Graham's method).
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img  # Image is too dark, return original
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to LAB color space
    or single channel image to enhance micro-lesion visibility.
    """
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def isolate_green_channel(img):
    """
    Extracts the green channel of the fundus image.
    The green channel offers maximum contrast for blood vessels, microaneurysms, and hemorrhages.
    """
    if img.ndim == 3:
        return img[:, :, 1]
    return img

def enhance_vessels(green_channel):
    """
    Enhances retinal blood vessels using Morphological Operations and Top-Hat filtering.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top_hat = cv2.morphologyEx(green_channel, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(green_channel, cv2.MORPH_BLACKHAT, kernel)
    
    enhanced = cv2.add(green_channel, top_hat)
    enhanced = cv2.subtract(enhanced, black_hat)
    return enhanced

def preprocess_fundus_ben_graham(img, target_size=512, sigmaX=10):
    """
    Complete Ben Graham Preprocessing Pipeline (APTOS 2019 Winning Approach):
    1. Crop dark borders
    2. Resize to square
    3. Gaussian blur & weighted subtraction to balance lighting
    """
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (target_size, target_size))
    
    # Gaussian blur subtraction for local illumination normalization
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    return enhanced

def process_fundus_pipeline(image_path, target_size=512):
    """
    Full pipeline returning dictionary of processed variants for clinical visual inspection.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = crop_image_from_gray(img_rgb)
    resized = cv2.resize(cropped, (target_size, target_size))
    
    green = isolate_green_channel(resized)
    clahe_green = apply_clahe(green, clip_limit=3.0)
    clahe_rgb = apply_clahe(resized, clip_limit=2.5)
    vessels = enhance_vessels(clahe_green)
    ben_graham = preprocess_fundus_ben_graham(resized, target_size=target_size)
    
    return {
        "original_cropped": resized,
        "green_channel": green,
        "clahe_rgb": clahe_rgb,
        "clahe_green": clahe_green,
        "vessels_enhanced": vessels,
        "ben_graham": ben_graham
    }

if __name__ == "__main__":
    print("Diabetic Retinopathy CV Preprocessing Module Loaded Successfully.")
