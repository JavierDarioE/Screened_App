import cv2
import os
import numpy as np

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

def classify_video(video, model):
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
    prediction_mean = np.mean(pred)
    if prediction_mean > 0.5:
        predicted_class = 'Super Mario Bros'  # Clase 1 (si el valor de predicción es mayor a 0.5)
    else:
        predicted_class = 'John Wick' 

    return predicted_class