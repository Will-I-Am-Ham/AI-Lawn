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
print(class_names)


# train_generator=train_datagen.flow_from_dataframe(
#                 dataframe=df,
#                 directory=path_to_GrassDataSet + "\\biomass_data\\train\\images",
#                 x_col="image",
#                 y_col="labels",
#                 subset="training",
#                 batch_size=32,
#                 seed=42,
#                 shuffle=True,
#                 class_mode="categorical",
#                 target_size=(32,32))

# valid_generator=train_datagen.flow_from_dataframe(
#                 dataframe=df,
#                 directory=path_to_GrassDataSet + "\\biomass_data\\train\\images",
#                 x_col="image",
#                 y_col="labels",
#                 subset="validation",
#                 batch_size=32,
#                 seed=42,
#                 shuffle=True,
#                 class_mode="categorical",
#                 target_size=(32,32))

# test_datagen=ImageDataGenerator(rescale=1./255.)

# test_generator=test_datagen.flow_from_dataframe(
#                 dataframe=testdf,
#                 directory=path_to_GrassDataSet + "\\biomass_data\\test\\images",
#                 x_col="image",
#                 y_col=None,
#                 batch_size=32,
#                 seed=42,
#                 shuffle=False,
#                 class_mode=None,
#                 target_size=(32,32))



# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(32, 32, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(17)
# ])


###################################Start changing stuff here###########################


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

model.summary()

epochs = 100

#####Dont Change this##############################################
# STEP_SIZE_TRAIN=int(train_generator.n/train_generator.batch_size)
# STEP_SIZE_VALID=int(valid_generator.n/valid_generator.batch_size)
# #STEP_SIZE_TEST=int(test_generator.n/test_generator.batch_size)

checkpoint_path = "training_1/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,
history= model.fit(train_generator,epochs=epochs, callbacks=[cp_callback])
###################################################################



#Training visualization
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history.history['loss']
#val_loss = history.history['val_loss']

epochs_range = range(epochs)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#model.evaluate(valid_generator,steps=STEP_SIZE_TEST)


#Predicting custom image

img = tf.keras.utils.load_img(
    '2.jpg', target_size=(32, 32)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#predicting a set amount of images
# test_generator.reset()
# pred=model.predict(test_generator,
# steps=STEP_SIZE_TEST,
# verbose=1)

# predicted_class_indices=np.argmax(pred,axis=1)

# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]

# filenames=test_generator.filenames

# print(predictions)

# if len(filenames) != len(predictions): 
#     print("badddd")
#     # Append mean values to the list with smaller length 
#     if len(filenames) > len(predictions): 
#         print("more files")
#         mean_width = st.mean(predictions) 
#         predictions += (len(filenames)-len(predictions)) * [mean_width] 
#     elif len(filenames) < len(predictions): 
#         print("more predictions")
#         mean_length = st.mean(filenames) 
#         filenames += (len(predictions)-len(filenames)) * [mean_length]

# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})
# results.to_csv("results.csv",index=False)