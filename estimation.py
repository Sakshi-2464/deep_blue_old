import cv2
import argparse
import torch
import numpy as np
from torch import nn, optim

# Your EstimatorModel class definition
class EstimatorModel(nn.Module):
    def __init__(self):
        super(EstimatorModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input layer (Age and Gender)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # Output layer (Height, Weight, BMI)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Load the pre-trained model weights
model = EstimatorModel()
model.load_state_dict(torch.load('models/height_weight_bmi_model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to detect faces
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frameOpencvDnn, faceBoxes

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to input image/video. Use 0 for webcam.", default=0)
args = parser.parse_args()

# Model files and configuration
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the pre-trained models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Open video capture (webcam or image/video file)
video = cv2.VideoCapture(args.image if args.image else 0)

# Padding around the detected face
padding = 20

# Define age range midpoints (average of each range)
ageMidpoints = [1, 5, 10, 17, 28.5, 40.5, 50.5, 80]

# Variable to store the last result
last_result = None

while True:
    hasFrame, frame = video.read()
    
    if not hasFrame:
        print("No frame detected or end of video stream.")
        break
    
    # Detect faces
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    for faceBox in faceBoxes:
        # Extract face ROI (Region of Interest)
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1), 
                     max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        # Predict gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predict age as continuous value
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = np.dot(agePreds[0], np.array(ageMidpoints))  # Predict exact age based on midpoints
        
        # Convert to integer (round it to the nearest integer)
        age = int(round(age))  # Convert to nearest integer
        
        # Prepare input for EstimatorModel (age and gender)
        gender_value = 1 if gender == 'Male' else 0  # Convert gender to binary (1 = Male, 0 = Female)
        model_input = torch.tensor([[age, gender_value]], dtype=torch.float32)
        
        # Predict height, weight, and BMI
        with torch.no_grad():
            prediction = model(model_input)
        
        # Extract the predicted height, weight, and BMI
        predicted_height, predicted_weight, predicted_bmi = prediction[0].numpy()
        
        # Update the last result
        last_result = {
            'Gender': gender,
            'Age': age,
            'Height': predicted_height,
            'Weight': predicted_weight,
            'BMI': predicted_bmi
        }
        
        # Display the results on the image
        cv2.putText(resultImg, f'Gender: {gender}, Age: {age} years', 
                    (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Predicted Height: {predicted_height:.2f} cm', 
                    (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Predicted Weight: {predicted_weight:.2f} kg', 
                    (faceBox[0], faceBox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Predicted BMI: {predicted_bmi:.2f}', 
                    (faceBox[0], faceBox[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 255), 2, cv2.LINE_AA)
        
    # Show the frame with detected face(s) and information
    cv2.imshow("Detecting Age, Gender, Height, Weight, and BMI", resultImg)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Print the last result after exiting
if last_result:
    print("\nFinal Result:")
    print(f"Gender: {last_result['Gender']}, Age: {last_result['Age']} years")
    print(f"Predicted Height: {last_result['Height']:.2f} cm")
    print(f"Predicted Weight: {last_result['Weight']:.2f} kg")
    print(f"Predicted BMI: {last_result['BMI']:.2f}")
else:
    print("No face detected during the session.")

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
