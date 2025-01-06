from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('attr_model.h5')

# Function to predict attributes from an image
def predict_attributes(image_path):
    # Load and preprocess the image (resize to 224x224 and normalize)
    img = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension (model expects input of shape (batch_size, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the attributes (age, height, weight, length, breadth, width)
    predictions = model.predict(img_array)

    # Get the predicted values
    predicted_age = int(predictions[0][5])  # Assuming age is the last attribute
    predicted_height = predictions[0][0]
    predicted_weight = predictions[0][1]
    predicted_length = predictions[0][2]
    predicted_breadth = predictions[0][3]
    predicted_width = predictions[0][4]

    # Output the results
    result = {
        "Age": predicted_age,
        "Height": predicted_height,
        "Weight": predicted_weight,
        "Length": predicted_length,
        "Breadth": predicted_breadth,
        "Width": predicted_width
    }

    return result

# Example usage:
image_path = 'C:/Users/tupka/Downloads/test.jpg'  # Replace with the path to your input image
predicted_attributes = predict_attributes(image_path)

# Print the results
print("Predicted Attributes:")
print(f"Age: {predicted_attributes['Age']} years")
print(f"Height: {predicted_attributes['Height']} cm")
print(f"Weight: {predicted_attributes['Weight']} kg")
print(f"Length: {predicted_attributes['Length']} cm")
print(f"Breadth: {predicted_attributes['Breadth']} cm")
print(f"Width: {predicted_attributes['Width']} cm")
