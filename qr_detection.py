from roboflow import Roboflow
import cv2
import json
import os

SECRET_FILE = "secret.txt"

if not os.path.exists(SECRET_FILE):
    raise FileNotFoundError("⚠️ ERROR: 'secret.txt' not found. Create it and add your API key.")

with open(SECRET_FILE, "r") as file:
    API_KEY = file.readline().strip()

if not API_KEY:
    raise ValueError("⚠️ ERROR: API key is empty. Add your Roboflow API key to 'secret.txt'.")

# Initialize Roboflow API
rf = Roboflow(api_key=API_KEY)  # Replace with your actual API key
project = rf.workspace().project("qr-code-authentication-mg5um")  # Replace with your project name
model = project.version(1).model  # Change '1' to the correct model version

# Load and process an image
image_path = os.path.abspath("generated_qr.png")

# Ensure the file exists before proceeding
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit()

result = model.predict(image_path, confidence=40, overlap=30).json()

# Display the results
print("Prediction Results:", result)

# Load image using OpenCV
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    print("Error: Unable to load the image. Check the file path.")
    exit()

# Extract predictions
for pred in result["predictions"]:
    x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
    class_name = pred["class"]
    
    # Draw bounding box
    cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
    
    # Put text label (Ensure it is inside the image)
    text_x = max(x - w//2, 0)
    text_y = max(y - h//2 - 10, 20)  # Avoid out-of-bounds text
    cv2.putText(image, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
output_path = "detection_result.png"
cv2.imwrite(output_path, image)
print(f"Detection result saved as '{output_path}'. Open the file to see the result.")

# Display the output image (Fix OpenCV display issue)
os.startfile(output_path)
