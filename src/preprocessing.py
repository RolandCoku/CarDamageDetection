import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
# We resize images to 64x64 pixels to keep training fast for the coursework.
# You can increase this to 128x128 later if you want better accuracy.
IMG_HEIGHT = 224
IMG_WIDTH = 224


def load_data(data_dir):
    """
    Loads images from the '00-damage' and '01-whole' folders,
    resizes them, and returns them as numpy arrays.
    """
    images = []
    labels = []

    # Define paths
    base_path = Path(data_dir)
    damage_path = base_path / '00-damage'
    whole_path = base_path / '01-whole'

    # Check if folders exist
    if not damage_path.exists() or not whole_path.exists():
        raise FileNotFoundError(f"Could not find folders in {data_dir}. Check your structure!")

    print(f"Loading images from {data_dir}...")

    # Load Damaged Cars (Label = 1)
    # We look for common image formats
    for img_file in damage_path.glob('*.[jJ][pP]*[gG]'):  # Matches .jpg, .jpeg, .JPG
        img = process_image(img_file)
        if img is not None:
            images.append(img)
            labels.append(1)  # 1 represents "Damaged"

    # Load Whole Cars (Label = 0)
    for img_file in whole_path.glob('*.[jJ][pP]*[gG]'):
        img = process_image(img_file)
        if img is not None:
            images.append(img)
            labels.append(0)  # 0 represents "Whole"

    return np.array(images), np.array(labels)


def process_image(file_path):
    """
    Reads an image file, resizes it, and normalizes pixel values.
    """
    try:
        # Read image in color
        img = cv2.imread(str(file_path))

        if img is None:
            return None

        # Resize to fixed dimensions (required for Neural Networks)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize pixel values to be between 0 and 1
        # (RGB values are 0-255, so we divide by 255.0)
        img = img / 255.0

        return img
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the function to see if it works
    # We assume 'data/training' is where you put the folders from your screenshot
    train_x, train_y = load_data('../data/training')
    print(f"Successfully loaded {len(train_x)} images.")
    print(f"Data shape: {train_x.shape}")