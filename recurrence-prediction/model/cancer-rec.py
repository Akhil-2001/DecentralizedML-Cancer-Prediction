############################################################################
## (C)Copyright 2021,2022 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import recall_score, f1_score

import logging
import os

import numpy as np
from six.moves import cPickle
from swarmlearning.tf import SwarmCallback
import sys


default_max_epochs = 100
default_min_peers = 2
batch_size = 30
num_classes = 2

data_dir = os.getenv('DATA_DIR', '/platform/swarmml/data')
model_dir = os.getenv('MODEL_DIR', '/platform/swarmml/model')
epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))

save_dir = os.path.join(model_dir, 'saved_models')
model_name = 'cancer-rec.h5'

# The data, split between train and test sets:
# Refer - https://keras.io/api/datasets/cifar10/

col_names=['id', 'outcome', 'time', 'radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean',	'smoothness_mean',	'compactness_mean',	'concavity_mean',	'concave_points_mean',	'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',	'perimeter_worst',	'area_worst',	'smoothness_worst', 'compactness_worst',	'concavity_worst',	'concave_points_worst',	'symmetry_worst',	'fractal_dimension_worst', 'tumor_size', 'lymph_node_status'	]

temp_df = pd.read_csv('https://raw.githubusercontent.com/Akhil-2001/Recurrance_Prediction/main/wpbc.data', skiprows=36, header=None, delimiter=',', skip_blank_lines=False)

temp_df.columns=col_names
df = temp_df

df['lymph_node_status'] = LabelEncoder().fit_transform(df['lymph_node_status'].values)

X = df.iloc[:,2:].values

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
X = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5','principal component 6','principal component 7','principal component 8','principal component 9','principal component 10'])


y = df.iloc[:,1].values

labelencode = LabelEncoder()
y=labelencode.fit_transform(y)

df['outcome']=y=labelencode.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)



# ADD YOUR MODEL CODE HERE

model1 = tf.keras.models.Sequential()

model1.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (10,)))
#model1.add(tf.keras.layers.Dropout(0.1))
model1.add(tf.keras.layers.Dense(units = 50, activation = 'relu'))
#model1.add(tf.keras.layers.Dropout(0.1))
model1.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


red = model1.predict(X_test)
mean=sum(red) / len(red)
for i in range(len(red)):
  if red[i] >= mean:
    red[i] = 1
  else:
    red[i] = 0

print('--------------------------------------------------------------')
print('Model Summary:')
print(model1.summary())
print('--------------------------------------------------------------')

# Let's train the model
model1.compile(optimizer = 'adagrad', loss ='binary_crossentropy', metrics=[tf.metrics.Recall(),'accuracy'])

# Create SwarmCallback
swarmCallback = SwarmCallback(syncFrequency=128,
                              minPeers=min_peers,
                              useAdaptiveSync=False,
                              adsValData=(X_test, y_test),
                              adsValBatchSize=batch_size)
swarmCallback.logger.setLevel(logging.DEBUG)

# Add SwarmCallback during training
model1.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True,
          callbacks=[swarmCallback])

# Save model and weights
swarmCallback.logger.info('Saving the final Swarm model ...')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model1.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

# Inference
swarmCallback.logger.info('Starting inference on the test data ...')
loss, recall, acc = model1.evaluate(X_test, y_test, verbose=1)
rec = recall_score(y_test,red)
f1 = f1_score(y_test,red)
swarmCallback.logger.info('Test loss = %.5f' % (loss))
swarmCallback.logger.info('Test accuracy = %.5f' % (acc))
swarmCallback.logger.info('Test recall = %.5f' % (rec))
swarmCallback.logger.info('Test recall = %.5f' % (f1))
