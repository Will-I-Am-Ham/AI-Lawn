import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

abalone_train = pd.read_csv(
    "GrassDataSet\\biomass_data\\train\\biomass_train_data.csv",
    names=["image_file_name", "acquisition_year", "seasonal_harvest_no", "label_type", 
           "fresh_grass", "dry_grass","fresh_white_clover","dry_white_clover","fresh_red_clover",
           "dry_red_clover","fresh_clover","dry_clover","fresh_weeds","dry_weeds","dry_total",
           "dry_clover_fraction","dry_red_clover_fraction","dry_white_clover_fraction","dry_grass_fraction","dry_weeds_fraction"],
    usecols=["fresh_grass", "dry_grass","fresh_white_clover","dry_white_clover","fresh_red_clover",
           "dry_red_clover","fresh_clover","dry_clover","fresh_weeds","dry_weeds","dry_total",
           "dry_clover_fraction","dry_red_clover_fraction","dry_white_clover_fraction","dry_grass_fraction","dry_weeds_fraction"])

print(abalone_train.head())

grass_features = abalone_train.copy()
grass_labels = grass_features.pop('dry_grass')
print(grass_labels)


grass_features = np.array(grass_features).astype(np.float32)
print(grass_features)

grass_model = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

grass_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

grass_model.fit(grass_features, grass_labels, epochs=10)
