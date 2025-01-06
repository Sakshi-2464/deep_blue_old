import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os

# Load the dataset
dataset_dir = './dataset/selfies'  # Directory containing images
csv_file_path = f"{dataset_dir}/selfies.csv"  # Path to the CSV file containing metadata
data = pd.read_csv(csv_file_path)

# Parameters
input_shape = (224, 224, 3)  # Image input size (224x224x3 for RGB images)
attributes = ['height', 'weight', 'length', 'breadth', 'width', 'age']  # Attributes to predict (including age)
num_attributes = len(attributes)  # Number of attributes

# Data preparation
images = []
attributes_data = []

for _, row in data.iterrows():
    try:
        # Construct the image path
        img_path = os.path.join(dataset_dir, row['image_path'])  # Make sure the 'image_path' column contains the relative path
        img = load_img(img_path, target_size=input_shape[:2])  # Resize image to (224, 224)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_array)

        # Collect the attributes (height, weight, length, breadth, width, age)
        attr_values = row[attributes].fillna(0).values  # Replace NaN with 0 for attributes
        attributes_data.append(attr_values)

    except Exception as e:
        print(f"Error loading {row['image_path']}: {e}")

# Convert to numpy arrays
images = np.array(images)
attributes_data = np.array(attributes_data)

# Split into train/test sets
X_images_train, X_images_val, X_attr_train, X_attr_val = train_test_split(
    images, attributes_data, test_size=0.2, random_state=42)

# Model definition (Short name: attr_model)
image_input = Input(shape=input_shape, name='image_input')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

x = Flatten()(x)

# Combine image features with final dense layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer for all attributes (regression outputs for height, weight, length, breadth, width, age)
output = Dense(num_attributes, activation='linear', name='attributes_output')(x)

# Define the model
attr_model = Model(inputs=image_input, outputs=output)

# Compile the model
attr_model.compile(optimizer='adam',
                   loss='mean_squared_error',  # Use MSE for regression
                   metrics=['mae'])  # MAE: Mean Absolute Error (alternative metric)

# Train the model
history = attr_model.fit(
    X_images_train, X_attr_train,
    validation_data=(X_images_val, X_attr_val),
    epochs=10,
    batch_size=16
)

# Save the model
attr_model.save('attr_model.h5')

print("Model training complete and saved as 'attr_model.h5'")

