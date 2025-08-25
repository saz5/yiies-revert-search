import os
from flask_cors import CORS
from towhee import pipe, ops
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)\
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Flask REST API!'})

# @app.route('/health', methods=['GET'])
# def echo():
#     data = request.get_json()
#     return jsonify({'you_sent': data}), 200


# Endpoint to receive a video file
@app.route('/create-video-embeddings', methods=['POST'])
def upload_video():
    post_id = request.form.get('post_id')
    if not post_id:
        return jsonify({'error': 'Post_id not specified'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'File no found'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'File without name'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Not allowed file type'}), 400
    
    filename = secure_filename(f"{'post_id'}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    query_pipe = (
        pipe.input('path')
            .map('path', 'frames', ops.video_decode.ffmpeg(start_time=None, end_time=None, sample_type='time_step_sample', args={'time_step': 0.5}))
            .map('frames', ('labels', 'scores', 'features'), ops.action_classification.pytorchvideo(model_name='x3d_m', skip_preprocess=True))
            .map('features', 'features', lambda features: features.tolist() )
            .output('features')
    )

    result = query_pipe( os.path.abspath(filepath) )

    # Delete the uploaded file after processing
    os.remove(filepath)
    result_data = result.get()[0]
    return jsonify({'embedding': result_data}), 200


if __name__ == '__main__':
    app.run( debug=True, port=5001 )
