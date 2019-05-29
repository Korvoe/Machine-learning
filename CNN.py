import pandas as pd
import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import to_categorical

#Data
data = pd.read_csv("emnist-letters-train.csv")
train_images = np.array(data.iloc[:60000,1:])
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = np.array(data.iloc[60000:80000,1:])
test_images = test_images.reshape(20000, 28, 28, 1)
train_labels = np.array(data.iloc[:60000,0])
test_labels = np.array(data.iloc[60000:80000,0])
train_images = train_images/255.0
test_images = test_images/255.0
train_labels = train_labels-1
test_labels = test_labels-1
train_labels = to_categorical(train_labels, num_classes=26)
test_labels = to_categorical(test_labels, num_classes=26)


#Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=12)
model.save('CNN_model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc*100, "%")
