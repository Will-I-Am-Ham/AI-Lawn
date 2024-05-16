from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
import pathlib
import os

import tensorflow as tf

import statistics as st 
from tensorflow.keras import layers



###############CHANGE THIS############
path_to_GrassDataSet = "C:\\Users\\meow1\\Documents\\CodingProjects\\GrassDataSet"
######################################

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
data_dir = pathlib.Path(path_to_GrassDataSet + "\\biomass_data\\train")
df = pd.read_csv(path_to_GrassDataSet + "\\biomass_data\\train\\biomass_train_data.csv")
testdf=pd.read_csv(path_to_GrassDataSet + "\\biomass_data\\test\\biomass_test_data.csv")
df["labels"] = df[["dry_total"]].round(-1).values.tolist()
testdf["labels"] = testdf[["dry_total"]].round(-1).values.tolist()
train_datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.4)
train_generator=train_datagen.flow_from_dataframe(
                dataframe=df,
                directory=path_to_GrassDataSet + "\\biomass_data\\train\\images",
                x_col="image",
                y_col="labels",
                subset="training",
                batch_size=32,
                seed=42,
                shuffle=True,
                class_mode="categorical",
                target_size=(32,32))


#make sure that model is the same!!! as where the weights came from!!#
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
model.add(Dense(17, activation='softmax'))


from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001, weight_decay = 0.000001), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()



##weight loading (Change this if you want ti change the weights)##
savedModel = model.load_weights('200.weights.h5')
###################################################################

print('Model Loaded!')


def predictdryness(inputPath = "input.jpg"):
  img = tf.keras.utils.load_img(
      inputPath, target_size=(32, 32)
  )

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  labels = (train_generator.class_indices)
  labels = dict((v,k) for k,v in labels.items())

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(labels[np.argmax(score)], 100 * np.max(score))
  )

predictdryness()