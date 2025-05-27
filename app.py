from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app, origins=["*"])

# Global model variable
model_session = None

def load_model():
    """Load YOLO11 ONNX model"""
    global model_session
    try:
        possible_paths = [
            "models/hi.onnx",
            "hi.onnx",
            os.path.join(os.getcwd(), "models", "hi.onnx"),
            os.path.join(os.getcwd(), "hi.onnx"),
            "/opt/render/project/src/models/hi.onnx",  # Render path
            "/opt/render/project/src/hi.onnx"          # Render path
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ Model file not found. Available files:")
            for root, dirs, files in os.walk(os.getcwd()):
                for file in files:
                    if file.endswith('.onnx'):
                        print(f"Found ONNX file: {os.path.join(root, file)}")
            raise FileNotFoundError("YOLO model file not found")
        
        print(f"📦 Loading model from: {model_path}")
        
        # Configure ONNX Runtime for Render (CPU-only)
        model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        print("✅ YOLO11 model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for YOLO11 inference"""
    input_size = 640
    
    # Resize maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    img_resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    img_padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    dx = (input_size - new_w) // 2
    dy = (input_size - new_h) // 2
    
    # Place resized image in center
    img_padded[dy:dy+new_h, dx:dx+new_w] = img_resized
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] and transpose to CHW format
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch

def postprocess_detections(outputs, conf_threshold=0.3):
    """Process YOLO11 outputs - SIMPLIFIED: Only confidence scores"""
    confidences = []
    
    # Handle different output formats
    predictions = outputs[0]
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    # YOLO11 output format: [num_detections, 84] or [84, num_detections]
    if predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T  # Transpose if needed
    
    print(f"🔍 Processing {predictions.shape[0]} predictions")
    
    for detection in predictions:
        if len(detection) >= 5:
            confidence = detection[4]  # Extract only confidence score
            
            if confidence > conf_threshold:
                confidences.append(float(confidence))
    
    print(f"🎯 Found {len(confidences)} valid detections")
    return confidences

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for health check"""
    return jsonify({
        "service": "YOLO11 Bin Detection API",
        "status": "running",
        "model_loaded": model_session is not None,
        "version": "1.0.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_session is not None,
        "service": "YOLO11 Bin Detection API",
        "version": "1.0.0",
        "environment": "production"
    })

@app.route('/detect', methods=['POST'])
def detect_bin():
    """Main detection endpoint - SIMPLIFIED RESPONSE"""
    try:
        # Check if model is loaded
        if model_session is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded",
                "message": "YOLO model is not available"
            }), 500
        
        # Get request data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image provided",
                "message": "Request must include base64 image data"
            }), 400
        
        image_b64 = data['image']
        confidence_threshold = data.get('threshold', 0.3)
        
        print(f"🔍 Processing detection request...")
        
        # Decode base64 image
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        image_data = base64.b64decode(image_b64)
        image_pil = Image.open(BytesIO(image_data))
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        original_height, original_width = image_cv.shape[:2]
        
        print(f"📐 Image size: {original_width}x{original_height}")
        
        # Preprocess image
        input_tensor = preprocess_image(image_cv)
        
        # Run inference
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: input_tensor})
        
        print("🧠 YOLO inference completed")
        
        # Get confidence scores only
        confidences = postprocess_detections(outputs, confidence_threshold)
        
        # Calculate confidence (as requested)
        if confidences:
            lowest_confidence = min(confidences)
            total_detections = len(confidences)
        else:
            lowest_confidence = 0.0
            total_detections = 0
        
        # Business logic: 75% minimum threshold for next step
        is_bin = lowest_confidence >= 0.75
        
        # Create response message (removed "lowest" word as requested)
        if lowest_confidence < 0.4:
            message = f"No bin detected (confidence: {lowest_confidence:.1%})"
        elif lowest_confidence < 0.75:
            message = f"Bin detected but confidence too low ({lowest_confidence:.1%}). Need 75% minimum to proceed."
        else:
            message = f"Bin detected with {lowest_confidence:.1%} confidence - Ready for next step!"
        
        # SIMPLIFIED RESPONSE
        response = {
            "success": True,
            "isBin": is_bin,
            "lowestConfidence": float(lowest_confidence),
            "message": message,
            "totalDetections": total_detections
        }
        
        print(f"✅ Detection completed: {total_detections} detections, confidence: {lowest_confidence:.1%}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Detection error: {error_msg}")
        return jsonify({
            "success": False,
            "error": error_msg,
            "message": "Detection failed",
            "isBin": False,
            "lowestConfidence": 0,
            "totalDetections": 0
        }), 500

# Initialize model when module loads (for Render)
def initialize_app():
    """Initialize the application"""
    print("🚀 Initializing YOLO11 Detection API for Render...")
    if not load_model():
        print("⚠️  Server started but model failed to load!")
        return False
    return True

# Initialize on import for production
initialize_app()

if __name__ == '__main__':
    print("🎯 Starting YOLO11 Bin Detection API...")
    
    # Get port from environment (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🌟 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)