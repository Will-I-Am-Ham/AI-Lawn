from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pathlib

import tensorflow as tf

###############CHANGE THIS############
path_to_DataSet = "C:\\Users\\meow1\\Documents\\CodingProjects\\Soil types"
######################################

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

train_datagen = ImageDataGenerator(rescale=1./255.,validation_split=0)
data_dir = pathlib.Path(path_to_DataSet).with_suffix('')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

train_generator = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(32, 32),
  batch_size=32)

valid_generator = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(32, 32),
  batch_size=32)
  

class_names = train_generator.class_names


#make sure that model is the same!!! as where the weights came from!!#

num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))


model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


from tensorflow.keras.optimizers import Adam

# model.compile(optimizer=Adam(learning_rate=0.001, weight_decay = 0.000001), loss="categorical_crossentropy", metrics=["accuracy"])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


##weight loading (Change this if you want ti change the weights)##
savedModel = model.load_weights('SoilColor.weights.h5')
###################################################################

print('Model Loaded!')


def predictColor(inputPath = "2.jpg"):

  img = tf.keras.utils.load_img(
      inputPath, target_size=(32, 32)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )



predictColor()