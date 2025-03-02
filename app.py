from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
import torch
from utils import detect_faces, detect_age_gender, estimate_height_weight_bmi, detect_objects, process_camera

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            
            # Load image
            image = cv2.imread(filepath)
            
            # Detect Faces & Predict Attributes
            faces = detect_faces(image)
            humans = []
            for (x, y, w, h) in faces:
                face = image[y:y + h, x:x + w]
                age, gender = detect_age_gender(face)
                height, weight, bmi = estimate_height_weight_bmi(age, gender)
                
                humans.append({
                    "age": age,
                    "gender": "Male" if gender == 0 else "Female",
                    "height": height,
                    "weight": weight,
                })
            
            # Object Detection with YOLOv3
            detected_objects = detect_objects(filepath)
            
            return jsonify({"humans": humans, "objects": detected_objects})
        elif "camera" in request.form:
            process_camera()
            return jsonify({"message": "Camera processing started. Press 'q' to exit."})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
