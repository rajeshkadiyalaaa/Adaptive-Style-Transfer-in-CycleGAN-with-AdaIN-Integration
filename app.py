import os
import base64
import re
from io import BytesIO

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from models.adain import StyleEncoder
from models.adain_cycle_gan import AdaINGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Path to the trained AdaIN-CycleGAN checkpoint (netG_A weights)
CHECKPOINT_PATH = os.environ.get(
    'CHECKPOINT_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'checkpoints', 'adain_cyclegan', 'latest_net.pth')
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f'Trained checkpoint not found at {CHECKPOINT_PATH}. '
        f'Train the model with train.py or set the CHECKPOINT_PATH environment '
        f'variable to a valid checkpoint.'
    )

generator = AdaINGenerator(input_nc=3, output_nc=3, ngf=64, use_adain=True).to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
generator.load_state_dict(checkpoint['netG_A'])
generator.eval()

style_encoder = StyleEncoder().to(device)
style_encoder.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def preprocess_image(img_bytes):
    """Decode image bytes into a normalized tensor of shape (1, 3, 256, 256)."""
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def deprocess_image(tensor):
    """Convert a generator output tensor in [-1, 1] to a base64 PNG string."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = ((tensor + 1) / 2.0).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    img = Image.fromarray(array)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


def style_transfer(content_img_bytes, style_img_bytes, style_weight=1.0):
    """Apply style transfer with the trained AdaIN-CycleGAN generator.

    The style weight is applied in feature space inside the AdaIN layers
    (interpolation between content and style statistics), not as a pixel-space
    blend of the output images.
    """
    content_image = preprocess_image(content_img_bytes)
    style_image = preprocess_image(style_img_bytes)

    with torch.no_grad():
        style_features = style_encoder(style_image)
        output = generator(content_image, style_features, alpha=style_weight)

    return deprocess_image(output)


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
