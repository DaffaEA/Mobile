# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import time
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Store latest predictions and images
latest_requests = []  # list of dicts: {timestamp, image_base64, predictions}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"Received request - Content-Type: {request.content_type}")
        logger.info(f"Has files: {'image' in request.files}")
        logger.info(f"Is JSON: {request.is_json}")
        
        image = None
        
        # Try to get image from files first
        if 'image' in request.files:
            logger.info("Loading image from files")
            image_file = request.files['image']
            image = Image.open(image_file.stream)
        
        # Try to get image from JSON
        elif request.is_json:
            data = request.get_json()
            logger.info(f"JSON keys: {list(data.keys()) if data else 'None'}")
            
            if data and 'image_data' in data:
                logger.info("Loading image from JSON")
                image_data = data['image_data']
                
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                    logger.info("Removed base64 prefix")
                
                logger.info(f"Base64 string length: {len(image_data)}")
                
                try:
                    image_bytes = base64.b64decode(image_data)
                    logger.info(f"Decoded bytes length: {len(image_bytes)}")
                    image = Image.open(io.BytesIO(image_bytes))
                    logger.info(f"Image opened: {image.size}, {image.mode}")
                except Exception as e:
                    logger.error(f"Failed to decode/open image: {e}")
                    return jsonify({
                        'error': f'Failed to process image data: {str(e)}',
                        'success': False
                    }), 400
            else:
                logger.error("No image_data in JSON")
                return jsonify({
                    'error': 'No image_data field in JSON',
                    'success': False
                }), 400
        else:
            logger.error("No image source found")
            return jsonify({
                'error': 'No image provided. Send either file or JSON with image_data',
                'success': False
            }), 400
        
        if image is None:
            logger.error("Image is None after processing")
            return jsonify({
                'error': 'Failed to load image',
                'success': False
            }), 400
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB")
        
        # Run YOLO prediction
        logger.info("Running YOLO prediction...")
        image_np = np.array(image)
        results = model(image_np, conf=0.5)
        
        predictions = []
        annotated_image = None
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                logger.info(f"Found {len(result.boxes)} detections")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    predictions.append({
                        'label': label,
                        'confidence': conf,
                        'bbox': [int(x) for x in box.xyxy[0].tolist()]
                    })
                annotated_image = result.plot()
            else:
                logger.info("No detections found")
        
        # Encode annotated image as base64 for dashboard
        img_base64 = None
        if annotated_image is not None:
            pil_img = Image.fromarray(annotated_image)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Save to latest_requests
        latest_requests.insert(0, {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'image_base64': img_base64,
            'predictions': predictions
        })
        # Keep only latest 20
        latest_requests[:] = latest_requests[:20]
        
        response = {
            'success': True,
            'predictions': predictions,
            'top_prediction': max(predictions, key=lambda x: x['confidence']) if predictions else None,
            'total_detections': len(predictions),
            'image_result': f"data:image/png;base64,{img_base64}" if img_base64 else None
        }
        
        logger.info(f"Returning {len(predictions)} predictions")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Exception in predict: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', requests=latest_requests)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)