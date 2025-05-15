from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from process_video import process_video_file
from moviepy.editor import VideoFileClip

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        duration = VideoFileClip(save_path).duration
        if duration > 300:
            os.remove(save_path)
            return jsonify({'error': '视频超过5分钟限制'}), 400
    except Exception as e:
        return jsonify({'error': f'读取视频失败: {str(e)}'}), 500

    try:
        # 直接处理视频并获取结果
        result = process_video_file(save_path)
        return jsonify({'message': '处理成功', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': f'视频处理失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
