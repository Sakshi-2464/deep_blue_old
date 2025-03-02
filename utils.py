import torch
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Age & Gender Models (Caffe)
AGE_MODEL = "models/age_net.caffemodel"
AGE_PROTO = "models/age_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Male", "Female"]

# Load PyTorch Model for Height, Weight, BMI Prediction
class HeightWeightBMIModel(torch.nn.Module):
    def __init__(self):
        super(HeightWeightBMIModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load trained PyTorch model
checkpoint = torch.load("height_weight_bmi.pth")
model = HeightWeightBMIModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load scaler from training
scaler_y = checkpoint["scaler_y"]

# Detect Faces (MediaPipe)
def detect_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    faces = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            faces.append((x, y, w, h))

    return faces

# Detect Age & Gender
def detect_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_preds[0].argmax()

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_bucket = AGE_BUCKETS[age_preds[0].argmax()]
    
    age = int(age_bucket.strip("()").split("-")[0])
    return age, gender

# Estimate Height, Weight, BMI
def estimate_height_weight_bmi(age, gender):
    input_data = np.array([[age, gender]], dtype=np.float32)
    input_tensor = torch.tensor(input_data)

    with torch.no_grad():
        predicted_scaled = model(input_tensor).numpy()

    predicted_values = scaler_y.inverse_transform(predicted_scaled)

    height = round(float(predicted_values[0][0]), 2)
    weight = round(float(predicted_values[0][1]), 2)
    bmi = int(round(predicted_values[0][2]))

    return height, weight, bmi

# Object Detection with YOLOv3
def detect_objects(image_path):
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    coco_names_path = "coco.names"

    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(coco_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (
                    int(detection[0] * width),
                    int(detection[1] * height),
                    int(detection[2] * width),
                    int(detection[3] * height),
                )

                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    PIXELS_PER_CM = 37.8  # Conversion factor

    if len(indexes) > 0:
        for i in indexes.flatten():
            label = classes[class_ids[i]]
            confidence = round(confidences[i], 2)

            # Ignore "person"
            if label.lower() == "person":
                continue

            width_cm = round(boxes[i][2] / PIXELS_PER_CM, 2)
            height_cm = round(boxes[i][3] / PIXELS_PER_CM, 2)

            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "width": width_cm,
                "height": height_cm
            })

    return detected_objects

# Process Camera Input
def process_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
