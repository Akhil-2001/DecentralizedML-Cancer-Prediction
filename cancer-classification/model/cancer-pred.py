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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense


import logging
import os

import numpy as np
from six.moves import cPickle
from swarmlearning.tf import SwarmCallback
import sys


default_max_epochs = 150
default_min_peers = 2
batch_size = 30
num_classes = 10

data_dir = os.getenv('DATA_DIR', '/platform/swarmml/data')
model_dir = os.getenv('MODEL_DIR', '/platform/swarmml/model')
epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))

save_dir = os.path.join(model_dir, 'saved_models')
model_name = 'cancer-pred.h5'

# The data, split between train and test sets:
# Refer - https://keras.io/api/datasets/cifar10/

df=pd.read_csv("https://raw.githubusercontent.com/Akhil-2001/Recurrance_Prediction/main/data.csv")

df.drop("Unnamed: 32",axis=1,inplace=True)

X=df.iloc[:,2:].values
y=df.iloc[:,1].values

labelencode = LabelEncoder()
y=labelencode.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


classifier = Sequential()
# ADD YOUR MODEL CODE HERE

#adding the input and first hidden layer
classifier.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform',input_dim=30))
#adding second layer
classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
#adding the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

print('--------------------------------------------------------------')
print('Model Summary:')
print(classifier.summary())
print('--------------------------------------------------------------')

# Let's train the model
classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

# Create SwarmCallback
swarmCallback = SwarmCallback(syncFrequency=128,
                              minPeers=min_peers,
                              useAdaptiveSync=False,
                              adsValData=(X_test, y_test),
                              adsValBatchSize=batch_size)
swarmCallback.logger.setLevel(logging.DEBUG)

# Add SwarmCallback during training
classifier.fit(X_train, y_train,
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
classifier.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

# Inference
swarmCallback.logger.info('Starting inference on the test data ...')
loss, acc = classifier.evaluate(X_test, y_test, verbose=1)
swarmCallback.logger.info('Test loss = %.5f' % (loss))
swarmCallback.logger.info('Test accuracy = %.5f' % (acc))
