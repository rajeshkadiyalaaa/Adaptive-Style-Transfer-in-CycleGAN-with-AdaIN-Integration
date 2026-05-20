import os
import base64
import re
from io import BytesIO

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the TF-Hub style transfer model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
style_transfer_model = tf.saved_model.load(model_path)

# Cache the transform to avoid recreating it
TARGET_SIZE = (384, 384)

def preprocess_image(img_data, target_size=TARGET_SIZE):
    """Preprocesses images for the style transfer model.
    
    Args:
        img_data: Image data (base64 string, bytes, or PIL Image)
        target_size: Tuple of (height, width) for resize
    
    Returns:
        tf.Tensor: Preprocessed image tensor with batch dimension
    """
    # Handle different input formats
    if isinstance(img_data, str):
        # Base64 string
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data)).convert('RGB')
    elif isinstance(img_data, bytes):
        # Raw bytes
        img = Image.open(BytesIO(img_data)).convert('RGB')
    else:
        # Already a PIL Image
        img = img_data.convert('RGB') if hasattr(img_data, 'convert') else img_data
    
    # Resize image to the target size
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension
    
    return img_tensor

def deprocess_image(tensor):
    """Deprocess tensor to base64-encoded PNG bytes.
    
    Args:
        tensor: tf.Tensor output from model
    
    Returns:
        str: Base64-encoded PNG image string
    """
    # Remove batch dimension
    tensor = tf.squeeze(tensor, axis=0)
    
    # Ensure values are in [0, 1]
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    
    # Scale to [0, 255] and convert to uint8
    tensor = (tensor * 255).numpy().astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(tensor)
    
    # Save to bytes buffer
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG', optimize=True)
    img_bytes = img_byte_arr.getvalue()
    
    # Return base64 encoded string
    return base64.b64encode(img_bytes).decode('utf-8')

def style_transfer(content_img_data, style_img_data, style_weight=1.0):
    """Applies style transfer using the pre-trained TF-Hub model.
    
    Args:
        content_img_data: Content image (bytes, base64 string, or PIL Image)
        style_img_data: Style image (bytes, base64 string, or PIL Image)
        style_weight: Weight for blending content and styled output (0.0 to 1.0)
    
    Returns:
        str: Base64-encoded stylized image
    """
    # Load and preprocess images with consistent size
    content_image = preprocess_image(content_img_data, TARGET_SIZE)
    style_image = preprocess_image(style_img_data, TARGET_SIZE)
    
    # Perform style transfer
    stylized_image = style_transfer_model(content_image, style_image)
    
    # Apply style weight interpolation if needed
    if style_weight != 1.0:
        # Interpolate between content and stylized image based on style_weight
        stylized_image = tf.add(
            tf.multiply(content_image, (1.0 - style_weight)),
            tf.multiply(stylized_image, style_weight)
        )
        # Ensure the values are in valid range
        stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0)
    
    return deprocess_image(stylized_image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    """Handle style transfer request.
    
    Expected form data:
        - contentImage: File upload or base64 string (webcam)
        - styleImage: File upload
        - styleWeight: Float between 0.0 and 1.0 (default: 1.0)
    
    Returns:
        JSON response with stylized image or error message
    """
    try:
        # Get style weight parameter
        style_weight = float(request.form.get('styleWeight', 1.0))
        # Clamp style weight to valid range
        style_weight = max(0.0, min(1.0, style_weight))
        
        # Handle content image (file upload or webcam)
        if 'contentImage' in request.files and request.files['contentImage'].filename != '':
            content_image = request.files['contentImage'].read()
        else:
            # Handle base64 webcam image
            content_base64 = request.form.get('webcamImage')
            if not content_base64:
                return jsonify({"error": "No content image provided"}), 400
            # Remove the data URL prefix if present
            if 'data:image' in content_base64:
                content_base64 = re.sub(r'^data:image/.+;base64,', '', content_base64)
            try:
                content_image = base64.b64decode(content_base64)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 content image: {str(e)}"}), 400
        
        # Handle style image
        if 'styleImage' not in request.files or request.files['styleImage'].filename == '':
            return jsonify({"error": "No style image provided"}), 400
        style_image = request.files['styleImage'].read()
        
        # Generate styled image
        result_image = style_transfer(content_image, style_image, style_weight)
        
        return jsonify({'image_data': f'data:image/png;base64,{result_image}'})
    
    except ValueError as e:
        print(f"Value Error: {e}")
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use production-ready settings (set debug=True only for development)
    app.run(debug=False, threaded=True, use_reloader=False)
