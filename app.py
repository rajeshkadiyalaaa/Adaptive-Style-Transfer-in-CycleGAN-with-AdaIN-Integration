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

def preprocess_image(img_data, target_size):
    """Preprocesses images for the style transfer model."""
    if isinstance(img_data, (str, bytes)):  # If base64 or bytes data
        img = Image.open(BytesIO(img_data)).convert('RGB')
    else:  # If already a PIL Image
        img = img_data
    img = img.resize(target_size)  # Resize image to the target size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

def deprocess_image(tensor):
    """Deprocess tensor to bytes."""
    tensor = tf.squeeze(tensor, axis=0)  # Remove batch dimension
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)  # Ensure values are in [0, 1]
    tensor = (tensor * 255).numpy().astype(np.uint8)  # Scale to [0, 255]
    img = Image.fromarray(tensor)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def style_transfer(content_img_data, style_img_data, style_weight=1.0):
    """Applies style transfer using the pre-trained TF-Hub model."""
    # Load and preprocess the content and style images
    content_image = preprocess_image(content_img_data, (384, 384))
    style_image = preprocess_image(style_img_data, (256, 256))

    # Use average pooling to smooth style image
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

    # Perform style transfer
    outputs = style_transfer_model(content_image, style_image)
    stylized_image = outputs[0]

    # Apply style weight
    if style_weight != 1.0:
        # Interpolate between content and stylized image based on style_weight
        stylized_image = tf.add(
            tf.multiply(content_image, (1 - style_weight)),
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
    try:
        style_weight = float(request.form.get('styleWeight', 1.0))
        
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
                content_base64 = re.sub('^data:image/.+;base64,', '', content_base64)
            content_image = base64.b64decode(content_base64)

        # Handle style image
        if 'styleImage' not in request.files or request.files['styleImage'].filename == '':
            return jsonify({"error": "No style image provided"}), 400
        style_image = request.files['styleImage'].read()

        # Generate styled image
        result_image = style_transfer(content_image, style_image, style_weight)
        
        return jsonify({'image_data': f'data:image/png;base64,{result_image}'})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
