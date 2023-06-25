# -*- coding: utf-8 -*-
"""Modelo_Identificacion_Peliculas.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/135hb-CVyeDhdfl_GIBMF11AmEmeBZRw5

# Modelo de predicción para películas.

## Importe de librerías necesarias.
"""

pip install git+https://github.com/keras-team/keras-preprocessing.git

import os
import random
import pandas as pd
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from keras import layers
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator
from google.colab.patches import cv2_imshow

"""## Preparación de los datos."""

drive.mount('/content/drive')

base_dir = '/content/drive/Shareddrives/TrainingDataRNAEquipo10/Movies_Data/'
john_dir = os.path.join(base_dir, 'john_wick_4')
mario_dir = os.path.join(base_dir, 'super_mario_bros')

def train_test_valid_split(data):
  # Definimos el tamaño de cada conjunto (train, validación y prueba)
  train_size = int(0.8 * len(data))
  val_size = int(0.1 * len(data))
  test_size = len(data) - train_size - val_size

  # Mezclamos aleatoriamente la data
  random.shuffle(data)

  # Dividimos los índices en conjuntos de entrenamiento, validación y prueba
  train_data = data[:train_size]
  val_data = data[train_size:train_size+val_size]
  test_data = data[train_size+val_size:]

  return train_data, test_data, val_data

# Set up matplotlib fig, and size it to fit 4x4 pics
nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index = 100

train_john_fnames, test_john_fnames, val_john_fnames = train_test_valid_split(os.listdir(john_dir))
train_mario_fnames, test_mario_fnames, val_mario_fnames = train_test_valid_split(os.listdir(mario_dir))

next_mario_pix = [os.path.join(mario_dir, fname) 
                for fname in train_mario_fnames[ pic_index-8:pic_index] 
               ]

next_john_pix = [os.path.join(john_dir, fname) 
                for fname in train_john_fnames[ pic_index-8:pic_index]
               ]


for i, img_path in enumerate(next_mario_pix+next_john_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print(len(train_mario_fnames), len(test_mario_fnames), len(val_mario_fnames))
print(len(train_john_fnames), len(test_john_fnames), len(val_john_fnames))

train_mario_full_paths = [os.path.join(mario_dir, fname) for fname in train_mario_fnames] 
train_john_full_paths = [os.path.join(john_dir, fname) for fname in train_john_fnames] 
train_df = pd.DataFrame(train_mario_full_paths)
train_df["class"] = "mario"
train_df.rename(columns = {0:'filename'}, inplace = True)
#train_df = train_df.append(pd.DataFrame({"filename": train_john_full_paths, "class":["john_wick_4"]*len(train_john_full_paths)}))
train_df = pd.concat([train_df, pd.DataFrame({"filename": train_john_full_paths, "class":["john_wick_4"]*len(train_john_full_paths)})])
train_df

valid_mario_full_paths = [os.path.join(mario_dir, fname) for fname in val_mario_fnames] 
valid_john_full_paths = [os.path.join(john_dir, fname) for fname in val_john_fnames] 
valid_df = pd.DataFrame(valid_mario_full_paths)
valid_df["class"] = "mario"
valid_df.rename(columns = {0:'filename'}, inplace = True)
#valid_df = valid_df.append(pd.DataFrame({"filename": valid_john_full_paths, "class":["john_wick_4"]*len(valid_john_full_paths)}))
valid_df = pd.concat([valid_df, pd.DataFrame({"filename": valid_john_full_paths, "class":["john_wick_4"]*len(valid_john_full_paths)})])
valid_df

# crear un ImageDataGenerator con nuestros datos.
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_dataframe(train_df, batch_size = 20, class_mode = 'binary', target_size = (150, 150))
validation_generator = test_datagen.flow_from_dataframe(valid_df, batch_size = 20, class_mode = 'binary', target_size = (150, 150))

"""## Implementación del modelo de predicción"""

base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])

inc_history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)

model.save('/content/drive/Shareddrives/RNA y algoritmos bioinspirados/Trabajo 03 - Desarrollo del prototipo de negocio/Movies_Data/modelo_inception_last_test.h5')

"""## Predicción de imágenes

### Predicción de una imagen aleatoria.
"""

# Directorio que contiene las imágenes a predecir
ruta_imagen = '/content/drive/Shareddrives/RNA y algoritmos bioinspirados/Trabajo 03 - Desarrollo del prototipo de negocio/Movies_Data/pruebaInception'

# Cargar el modelo entrenado
model = tf.keras.models.load_model('/content/drive/Shareddrives/RNA y algoritmos bioinspirados/Trabajo 03 - Desarrollo del prototipo de negocio/Movies_Data/modelo_inception_last_test.h5')

# Obtener la lista de archivos en el directorio
lista_archivos = os.listdir(ruta_imagen)

# Iterar a través de los archivos
for filename in lista_archivos:
    # Obtener la ruta completa del archivo
    img_path = os.path.join(ruta_imagen, filename)
    image = load_img(img_path, target_size=(150, 150))
    image_array = img_to_array(image)  # Convierte la imagen a un arreglo numpy
    image_array = image_array / 255.0  # Normaliza los valores de los pixeles en el rango [0, 1] (igual que durante el entrenamiento)
    image_array = np.expand_dims(image_array, axis=0) 

    # Realizar la predicción
    prediction = model.predict(image_array)
    if prediction[0][0] > 0.5:
      predicted_class = 'Super Mario Bros'  # Clase 1 (si el valor de predicción es mayor a 0.5)
    else:
      predicted_class = 'John Wick' 

    # Mostrar la imagen y su etiqueta predicha
    plt.imshow(mpimg.imread(img_path))
    plt.title('Etiqueta predicha: {}'.format(predicted_class))
    plt.axis('off')
    plt.show()

"""### Predicción de imágenes representativas en un video."""

from google.colab import files
uploaded_video = files.upload()

print(list(uploaded_video.keys())[0])

model = tf.keras.models.load_model('/content/drive/Shareddrives/RNA y algoritmos bioinspirados/Trabajo 03 - Desarrollo del prototipo de negocio/Movies_Data/modelo_inception_last_test.h5')

#cargar el video.
cap = cv2.VideoCapture(list(uploaded_video.keys())[0])

# #obtener el numero total de frames en el video.
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# definir la longitud de intervalo entre la cual capturar los frames.
interval = num_frames // 5

# extraer los frames seleccionados
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

    # Mostrar la imagen y su etiqueta predicha
    plt.imshow(img)
    plt.title('Etiqueta predicha: {}'.format(predicted_class))
    plt.axis('off')
    plt.show()

"""### Predicción final del video."""

result = np.mean(pred)
if result > 0.5:
  print("Película: Super Mario Bros")
else:
  print("Película: John Wick 4")