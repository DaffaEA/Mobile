# test_predict.py
import requests
import base64
import json

# URL of your Flask server
SERVER_URL = "http://127.0.0.1:5000/predict"  # change to your server IP if needed

# Path to the image you want to test
IMAGE_PATH = "image.png"

def send_image(image_path):
    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Convert to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "image_data": f"data:image/jpeg;base64,{image_base64}"
    }

    # Send POST request
    response = requests.post(SERVER_URL, json=payload)
    return response

def main():
    response = send_image(IMAGE_PATH)
    if response.status_code == 200:
        data = response.json()
        print("Success:", data.get("success"))
        print("Total Detections:", data.get("total_detections"))
        print("Top Prediction:", data.get("top_prediction"))
        print("All Predictions:")
        for pred in data.get("predictions", []):
            print(f"  Label: {pred['label']}, Confidence: {pred['confidence']}, BBox: {pred['bbox']}")
        
        # Optionally save annotated image
        if "image_result" in data:
            img_base64 = data["image_result"].split("base64,")[1]
            img_bytes = base64.b64decode(img_base64)
            with open("annotated_result.png", "wb") as f:
                f.write(img_bytes)
            print("Annotated image saved as annotated_result.png")
    else:
        print("Request failed with status code:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    main()
