import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Specify the path to the dataset directory
dataset_dir = r'C:\Users\tupka\Downloads\Age-and-Gender-Detection-OpenCV--Keras--TensorFlow\dataset\selfies'
csv_file_path = os.path.join(dataset_dir, 'selfies.csv')

# Check if the dataset directory exists
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"The specified dataset directory does not exist: {dataset_dir}")

# Check if the CSV file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The metadata file (selfies.csv) does not exist: {csv_file_path}")

# Load the CSV file
metadata = pd.read_csv(csv_file_path)
print("CSV file loaded successfully!")
print(metadata.head())

# Ensure the required columns exist
required_columns = ['image_path', 'height', 'weight', 'age', 'length', 'breadth', 'width']
missing_columns = [col for col in required_columns if col not in metadata.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing in the CSV file: {missing_columns}")

# Verify that all image files listed in the CSV exist in the dataset directory
missing_images = []
for img_name in metadata['image_path']:
    img_path = os.path.join(dataset_dir, img_name)
    if not os.path.exists(img_path):
        missing_images.append(img_name)

if missing_images:
    print(f"Warning: The following images are listed in the CSV but not found in the dataset directory: {missing_images}")
else:
    print("All image files listed in the CSV are present!")

# Example: Load one sample image and its attributes for demonstration
sample_row = metadata.iloc[0]
sample_img_path = os.path.join(dataset_dir, sample_row['image_path'])

# Load the image
img = load_img(sample_img_path, target_size=(224, 224))  # Resize to 224x224
img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]

# Get the corresponding attributes
attributes = {
    'height': sample_row['height'],
    'weight': sample_row['weight'],
    'age': sample_row['age'],
    'length': sample_row['length'],
    'breadth': sample_row['breadth'],
    'width': sample_row['width'],
}

print(f"Sample Image Path: {sample_img_path}")
print(f"Image Shape: {img_array.shape}")
print(f"Attributes: {attributes}")

# Optional: Convert all images and attributes into a NumPy array for training
image_data = []
attribute_data = []

for _, row in metadata.iterrows():
    img_path = os.path.join(dataset_dir, row['image_path'])
    if os.path.exists(img_path):
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        image_data.append(img_array)
        
        # Load attributes (as an example, include height, weight, and age only)
        attribute_data.append([row['height'], row['weight'], row['age']])

# Convert lists to NumPy arrays
image_data = np.array(image_data)
attribute_data = np.array(attribute_data)

print(f"Image Data Shape: {image_data.shape}")
print(f"Attribute Data Shape: {attribute_data.shape}")
