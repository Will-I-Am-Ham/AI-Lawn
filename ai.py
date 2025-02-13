import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from tensorflow.keras import datasets, models

def analyis():
  (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
  training_images, testing_images = training_images/255, testing_images/255

  class_names = ['siltsoil','claysoil','sandysoil','loamysoil','rockysoil','drysoil','packedsoil']


  training_images = training_images[:20000]
  training_labels = training_labels[:20000]
  testing_images = testing_images[:4000]
  testing_labels = testing_labels[:4000]

  model = models.load_model('image_classifier.model')

  img = cv.imread('soilreal.png')
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

  plt.imshow(img, cmap=plt.cm.binary)

  prediction = model.predict(np.array([img]) / 255)
  index = np.argmax(prediction)
  print("Prediction is: {class_names[index]}")
  return class_names[index]

analyis()
