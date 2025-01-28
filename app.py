from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from cbir_resnet import extract_features, find_similar_images

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Ensure upload and result folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'query_image' not in request.files:
            return "No file part"
        file = request.files['query_image']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            query_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(query_image_path)

            # Find similar images
            results = find_similar_images(query_image_path, app.config['RESULT_FOLDER'])

            # Render results
            return render_template('index.html', query_image=url_for('static', filename=f'uploads/{filename}'), results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
