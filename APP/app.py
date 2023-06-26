import identification_module
import os
import urllib.request
from flask import Flask, render_template, request, url_for, redirect, flash
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('modelo_inception_last_test.h5')

@app.route('/')
def upload_form():
    return render_template('inicio.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Realizar la clasificación del video
        video_class = identification_module.classify_video(video_path, model)
        
        flash('El video ha sido identificado exitosamente.')
        flash('El nombre de la película es: '+ video_class)
        return render_template('inicio.html', filename=filename, video_class=video_class)

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run()
