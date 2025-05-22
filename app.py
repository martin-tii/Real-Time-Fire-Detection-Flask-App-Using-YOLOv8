from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import logging

# Suppress YOLO logs
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)

app = Flask(__name__, static_url_path='/static')
model = YOLO('best.pt')  # Change to your actual model path if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Detect from webcam and return a JPEG image"""
    result = model.predict(source=0, imgsz=640, conf=0.6, show=False)[0]
    img = result.plot()
    success, jpeg = cv2.imencode('.jpg', img)
    if not success:
        return Response(status=500)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/detect_meta', methods=['POST'])
def detect_meta():
    """Detect from webcam and return metadata (used for polling interfaces)"""
    result = model.predict(source=0, imgsz=640, conf=0.6, show=False)[0]
    has_fire = result.boxes is not None and len(result.boxes) > 0
    return jsonify({"fire": has_fire})

@app.route('/detect_from_image', methods=['POST'])
def detect_from_image():
    """Detect fire and return both prediction status and annotated image as base64."""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]  # Strip the header
        decoded = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        result = model.predict(source=img, imgsz=640, conf=0.6, show=False)[0]
        has_fire = result.boxes is not None and len(result.boxes) > 0

        # Annotate the image
        annotated_img = result.plot()
        _, jpeg = cv2.imencode('.jpg', annotated_img)
        base64_img = base64.b64encode(jpeg).decode('utf-8')
        base64_url = f"data:image/jpeg;base64,{base64_img}"

        return jsonify({
            "fire": has_fire,
            "num_detections": len(result.boxes) if result.boxes else 0,
            "image": base64_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9000)
