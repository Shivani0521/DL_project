import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import tensorflow as tf
import io
import os

app = Flask(__name__)
CORS(app)

# Static folder for saving output images
OUTPUT_DIR = os.path.join(os.getcwd(), "static")  # Local directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the model paths and disease labels for different models
MODEL_PATHS = {
    "mango": r"Mango.keras",  # Path to mango model
    "strawberry": r"Strawberry.keras"  # Path to strawberry model
}

DISEASE_LABELS = {
    "mango": [
        "Alternaria", "Anthracnose", "Bacterial Canker", "Black Mould Rot",
        "Cutting Weevil", "Die Back", "Gall Midge", "Healthy", "Powdery Mildew",
        "Scooty Mould", "Stem End Rot"
    ],
    "strawberry": [
        "Angular Leaf Spot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Healthy",
        "Leaf Spot", "Powdery Mildew"
    ]
}

# Load models into memory
models = {}
for model_type, model_path in MODEL_PATHS.items():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        models[model_type] = model
        print(f"Model for {model_type} loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {model_type}: {e}")

# Enhanced fruit color detection with round shape detection
def detect_fruit_color_and_shape(image, fruit_type):
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # Mango Color Range (Yellow to Orange)
    if fruit_type == "mango":
        # Expand the HSV range dynamically for mango colors (more yellow to orange)
        lower_fruit_mango = np.array([10, 50, 50])  # Lower bound for yellow/orange
        upper_fruit_mango = np.array([40, 255, 255])  # Upper bound for yellow/orange

        mask_fruit = cv2.inRange(hsv_image, lower_fruit_mango, upper_fruit_mango)

    # Strawberry Color Range (Red)
    elif fruit_type == "strawberry":
        # Dynamically adjust the HSV range for strawberry red color detection
        lower_fruit_strawberry = np.array([0, 50, 50])  # Lower bound for red
        upper_fruit_strawberry = np.array([10, 255, 255])  # Upper bound for red
        mask_fruit = cv2.inRange(hsv_image, lower_fruit_strawberry, upper_fruit_strawberry)

    # Find contours of the detected regions
    contours, _ = cv2.findContours(mask_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set a minimum area threshold for detected fruit regions
    min_area = 2000  # Adjust based on your image size and requirements

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Calculate the aspect ratio of the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / float(h)

            # To detect round shapes, the aspect ratio should be close to 1
            if 0.8 < aspect_ratio < 1.2:  # Allow a small tolerance for round shapes
                return True  # Detected fruit region with round shape

    return False  # No fruit region detected or it's too small/not round enough

@app.route('/')
def index():
    return render_template('index_modified.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the fruit type from request parameters (e.g., "mango" or "strawberry")
    fruit_type = request.form.get('fruit_type', '').lower()
    if fruit_type not in models:
        return jsonify({"error": "Invalid fruit type provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load the image from the file
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Check if the image contains the expected fruit color and round shape
        fruit_detected = detect_fruit_color_and_shape(img, fruit_type)

        if not fruit_detected:
            return jsonify({"error": f"No {fruit_type} detected in the image"}), 400

        original_width, original_height = img.size

        # Dynamically resize the image based on the selected fruit type
        if fruit_type == "mango":
            input_size = (224, 224)  # Mango model requires 224x224 input size
        elif fruit_type == "strawberry":
            input_size = (299, 299)  # Strawberry model requires 299x299 input size

        # Resize the image to match model input size
        img_resized = img.resize(input_size)
        img_array = np.array(img_resized) / 255.0  # Normalize the image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get the model and labels for the selected fruit
        model = models[fruit_type]
        disease_labels = DISEASE_LABELS[fruit_type]

        # Make a prediction using the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = disease_labels[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100

        # Mock bounding box coordinates (for illustration purposes)
        bbox_resized = [50, 50, 150, 150]  # Example coordinates
        # Scale the bounding box to the original image size
        x_min = int(bbox_resized[0] * (original_width / input_size[0]))
        y_min = int(bbox_resized[1] * (original_height / input_size[1]))
        x_max = int(bbox_resized[2] * (original_width / input_size[0]))
        y_max = int(bbox_resized[3] * (original_height / input_size[1]))
        bbox_original = [x_min, y_min, x_max, y_max]

        # Draw the bounding box and label on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox_original, outline="red", width=3)
        draw.text((bbox_original[0], bbox_original[1] - 10),
                  f"{predicted_label} ({confidence:.2f}%)", fill="red")

        # Save the processed image with bounding box to the output directory
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        img.save(output_path)

        # Respond with the prediction details and image URL
        return jsonify({
            "label": predicted_label,
            "confidence": f"{confidence:.2f}",
            "image_url": f"/static/{output_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=4000)