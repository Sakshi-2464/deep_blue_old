document.addEventListener("DOMContentLoaded", function () {
    let fileInput = document.getElementById("fileInput");
    let fileNameDisplay = document.getElementById("fileNameDisplay");
    let video = document.getElementById("cameraFeed");
    let captureButton = document.getElementById("captureButton");
    let analyzeButton = document.getElementById("analyzeButton");
    let canvas = document.createElement("canvas");
    let imagePreview = document.getElementById("imagePreview");
    let imagePreviewContainer = document.getElementById("image-preview-container");
    let stream = null;

    // Hide Capture button initially
    captureButton.style.display = "none";

    // Update file name when selecting an image
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
            analyzeButton.style.display = "inline-block"; // Show Analyze button

            // Show the selected image
            let reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = "block";
                imagePreviewContainer.style.display = "block";
            };
            reader.readAsDataURL(fileInput.files[0]);
        } else {
            fileNameDisplay.textContent = "No file chosen";
            analyzeButton.style.display = "none"; // Hide Analyze button
        }
    });

    // Start Camera
    window.startCamera = function () {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (cameraStream) {
                stream = cameraStream;
                video.srcObject = stream;
                video.style.display = "block";
                captureButton.style.display = "inline-block"; // Show Capture button
            })
            .catch(function (error) {
                console.error("Error accessing camera:", error);
                alert("Failed to access camera. Please check permissions.");
            });
    };

    // Capture Image from Camera
    window.capturePhoto = function () {
        if (!stream) {
            alert("Camera not started!");
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        let context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function (blob) {
            let formData = new FormData();
            formData.append("file", blob, "captured_image.jpg");

            // Convert Blob to Image Preview
            let reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = "block";
                imagePreviewContainer.style.display = "block";
            };
            reader.readAsDataURL(blob);

            sendImage(formData);

            // Stop Camera after capture
            stream.getTracks().forEach(track => track.stop());
            video.style.display = "none";
            captureButton.style.display = "none";
        }, "image/jpeg");
    };

    // Upload Image from File
    window.uploadImage = function () {
        if (fileInput.files.length === 0) {
            alert("Please select an image.");
            return;
        }
        let formData = new FormData();
        formData.append("file", fileInput.files[0]);
        sendImage(formData);
    };

    // Send Image for Analysis
    function sendImage(formData) {
        fetch("/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(displayResults)
        .catch(error => console.error("Error:", error));
    }

    // Display Analysis Results
    function displayResults(data) {
        let resultContainer = document.getElementById("result");
        resultContainer.innerHTML = "<h2>Results:</h2>"; // Clear previous results

        if (data.humans.length > 0) {
            let humanResults = document.createElement("div");
            humanResults.classList.add("result-grid");

            data.humans.forEach((human, index) => {
                let personCard = document.createElement("div");
                personCard.classList.add("result-card");

                personCard.innerHTML = `
                    <h3>Person ${index + 1}</h3>
                    <p><strong>Age:</strong> ${human.age}</p>
                    <p><strong>Gender:</strong> ${human.gender}</p>
                    <p><strong>Height:</strong> ${human.height} cm</p>
                    <p><strong>Weight:</strong> ${human.weight} kg</p>
                `;

                humanResults.appendChild(personCard);
            });

            resultContainer.appendChild(humanResults);
        } else {
            resultContainer.innerHTML += "<p>No human detected.</p>";
        }

        let filteredObjects = data.objects.filter(obj => obj.label.toLowerCase() !== "person");

        if (filteredObjects.length > 0) {
            let objectResults = document.createElement("div");
            objectResults.classList.add("result-grid");

            filteredObjects.forEach((obj, index) => {
                let objectCard = document.createElement("div");
                objectCard.classList.add("result-card");

                objectCard.innerHTML = `
                    <h3>Object ${index + 1}: ${obj.label}</h3>
                    <p><strong>Width:</strong> ${obj.width} cm</p>
                    <p><strong>Height:</strong> ${obj.height} cm</p>
                `;

                objectResults.appendChild(objectCard);
            });

            resultContainer.appendChild(objectResults);
        } else {
            resultContainer.innerHTML += "<p>No objects detected.</p>";
        }
    }
});
