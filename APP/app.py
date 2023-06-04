import os
import urllib.request
import cv2
from flask import Flask, render_template, request, url_for, redirect, flash
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('modelo_inception_last_test.h5')

def preprocess_video(video):
    # Cargar y preprocesar el fragmento de video utilizando las mismas transformaciones
    # y preprocesamientos aplicados durante el entrenamiento del modelo
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = num_frames // 5
    frames = []
    for i in range(1, 6):
        frame_idx = interval * i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
    cap.release()
    step = len(frames) // 5
    selected_frames = frames[step:step*5:step]
    cv2.destroyAllWindows()
    return selected_frames

def classify_video(video):
    # Preprocesar el fragmento de video
    selected_frames = preprocess_video(video)
    pred = []
    for image_array in selected_frames:
        img = image_array
        image_array = image_array / 255.0  # Normaliza los valores de los pixeles en el rango [0, 1] (igual que durante el entrenamiento)
        image_array = np.expand_dims(image_array, axis=0) 

        # Realizar la predicción
        prediction = model.predict(image_array)
        print(prediction)
        pred.append(prediction[0][0])
        if prediction[0][0] > 0.5:
            predicted_class = 'Super Mario Bros'  # Clase 1 (si el valor de predicción es mayor a 0.5)
        else:
            predicted_class = 'John Wick' 

    return predicted_class

@app.route('/')
def upload_form():
    return render_template('upload.html')

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
        video_class = classify_video(video_path)
        
        flash('Video successfully uploaded and displayed below')
        flash('The name of the movie is '+ video_class)
        return render_template('upload.html', filename=filename, video_class=video_class)

@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run()
