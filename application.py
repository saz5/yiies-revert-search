import os
import requests
from io import BytesIO
from flask_cors import CORS
from towhee import pipe, ops
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 Megabytes
CORS(application )

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = { 'png', 'jpeg', 'jpg' }
ALLOWED_VIDEO_EXTENSIONS = { 'mp4', 'avi', 'mov', 'mkv', 'webm' }
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def allowed_video_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def allowed_image_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@application.route('/', methods=['GET'])
@application.route('/health', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Flask REST API!'})

# Endpoint to receive a video file
@application.route('/create-video-embeddings', methods=['POST'])
def create_video_embeddings():
    post_id = request.form.get('post_id')
    if not post_id:
        return jsonify({'error': 'Post_id not specified'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'File no found'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'File without name'}), 400
    
    if not file or not allowed_video_format(file.filename):
        return jsonify({'error': 'Not allowed file type'}), 400

    filename = secure_filename(f"{post_id}_{file.filename}")
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    video_pipe = (
        pipe.input('path')
            .map('path', 'frames', ops.video_decode.ffmpeg(start_time=None, end_time=None, sample_type='time_step_sample', args={'time_step': 0.5}))
            .map('frames', ('labels', 'scores', 'features'), ops.action_classification.pytorchvideo(model_name='x3d_m', skip_preprocess=True))
            .map('features', 'features', lambda features: features.tolist() )
            .output('features')
    )

    result = video_pipe( os.path.abspath(filepath) )

    # Delete the uploaded file after processing
    os.remove(filepath)

    result_data = result.get()[0]
    return jsonify({ 'post_id': post_id, 'embedding': result_data }), 200

@application.route('/create-image-embeddings', methods=['POST'])
def create_image_embeddings():
    post_id = request.form.get('post_id')
    if not post_id:
        return jsonify({'error': 'Post_id not specified'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'File no found'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'File without name'}), 400
    
    if not file or not allowed_image_format(file.filename):
        return jsonify({'error': 'Not allowed file type'}), 400
    
    filename = secure_filename(f"{post_id}_{file.filename}")

    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img_pipe = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        # .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
        .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
        .map('vec', 'vec', lambda vec: vec.tolist() )
        .output('vec')
    )
    result = img_pipe(filepath)

    result_data = result.get()[0]

    os.remove(filepath)
    return jsonify({ 'post_id': post_id, 'embedding': result_data }), 200

@application.route('/create-image-embeddings-url', methods=['POST'])
def create_image_embeddings_url():
    print("CRETE_IMAGE_EMBEDDINGS_URL")
    data = request.get_json()
    post_id = data.get('post_id')
    image = data.get('image')

    print("POST_ID", post_id, "IMAGE", image)

    if not image:
        return "Missing 'url' parameter", 400

    try:
        # Fetch the file content from the external URL
        response = requests.get(image)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get the filename from the URL, or use a default
        filename = image.split('/')[-1]

        filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        img_pipe = (
            pipe.input('url')
            .map('url', 'img', ops.image_decode.cv2_rgb())
            # .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
            .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
            .map('vec', 'vec', lambda vec: vec.tolist() )
            .output('vec')
        )
        result = img_pipe(filepath)

        result_data = result.get()[0]

        os.remove(filepath)
        return jsonify({ 'post_id': post_id, 'embedding': result_data }), 200
    
    except requests.exceptions.RequestException as e:
        return jsonify({ "error": "Error downloading file"}), 500
    
@application.route('/create-video-embeddings-url', methods=['POST'])
def create_video_embeddings_url():
    data = request.get_json()
    post_id = data.get('post_id')
    video = data.get('video')

    if not video:
        return "Missing 'url' parameter", 400

    try:
        # Fetch the file content from the external URL
        response = requests.get(video)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get the filename from the URL, or use a default
        filename = video.split('/')[-1]

        filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(response.content)

        video_pipe = (
            pipe.input('path')
                .map('path', 'frames', ops.video_decode.ffmpeg(start_time=None, end_time=None, sample_type='time_step_sample', args={'time_step': 0.5}))
                .map('frames', ('labels', 'scores', 'features'), ops.action_classification.pytorchvideo(model_name='x3d_m', skip_preprocess=True))
                .map('features', 'features', lambda features: features.tolist() )
                .output('features')
        )

        result = video_pipe( os.path.abspath(filepath) )

        # Delete the uploaded file after processing
        os.remove(filepath)

        result_data = result.get()[0]
        return jsonify({ 'post_id': post_id, 'embedding': result_data }), 200
    
    except requests.exceptions.RequestException as e:
        return jsonify({ "error": "Error downloading file"}), 500

def test_create_image_embedding():
    img_pipe = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        # .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
        .map('img', 'vec', ops.image_embedding.timm(model_name='resnet50'))
        .output('vec')
    )
    result = img_pipe('./uploads/test.jpg')

    result.get()[0]

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8025)
    # application.run(host='localhost', port=5001, debug=True)
