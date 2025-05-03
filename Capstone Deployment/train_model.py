import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense


# ---------- #


def load_images_from_path(path, label):
    images = []
    labels = []


    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))
       
    return images, labels




keys = ['A', 'Ab', 'B', 'Bb', 'C', 'D', 'Db', 'E', 'Eb', 'F', 'G', 'Gb']


x = []
y = []


label_counter = 0


for k in keys:
   
    images, labels = load_images_from_path(f"../Data/processed/Chromagram/{k}", label_counter)
   
    x += images
    y += labels


    label_counter = label_counter + 1




x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)


x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255


y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)




# ----------- #




model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(12, activation='softmax'))




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = [
    tf.keras.callbacks.ModelCheckpoint('minor_model.keras', save_best_only=True)
]




hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=10, callbacks=callbacks)




test_loss, test_acc = model.evaluate(x_test_norm, y_test_encoded)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")