import tensorflow as tf
import cv2
import numpy as np
import os



model_dir = r'./model/keras_model.h5'
className_dir = r'./model/labels.txt'

model = tf.keras.models.load_model(model_dir, compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

class_names = \
    {
        '0': 'Sea Bass',
        '1': 'Red Sea Bream',
        '2': 'Red Mullet',
        '3': 'Hourse Mackerel',
    }

while True:

    img_dir = input(r'Input your image direction:')

    if img_dir == '0':
        break

    if '"' in img_dir:
        img_dir = img_dir.strip('"')

    if os.path.isfile(img_dir) == True:

        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = img.shape

        if shape != (224, 224, 3):
            img = cv2.resize(img, (224, 224))

        normalized_image_array = (img.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        idx = np.argmax(prediction)
        class_name = class_names[str(idx)]

        print(f"Class: {class_name}\n", class_name)


    else:

        print('Error Direction')

print('Program has been exited')