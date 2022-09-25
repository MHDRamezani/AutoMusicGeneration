"""
	This file trains a Seq2Seq LSTM model to learn to play music
"""

import sys
import os
import time
import ipykernel
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from midi_parser import getData, createTrainData

# GLOBAL PARAMETERS
highest_note = 81  # A_6 	Needs to be consistent with the value in midi_parser.py
lowest_note = 33  # A_2		Needs to be consistent with the value in midi_parser.py
pitch_dimension = highest_note - lowest_note + 1

# Model parameters
num_hidden = 512
x_length = 100
y_length = 10
batch_size = 64
num_epochs = 100


def buildModel():
    '''Build a Seq2Seq LSTM model'''

    # encoder
    model = Sequential()
    model.add(LSTM(num_hidden, input_dim=pitch_dimension, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(num_hidden))
    model.add(RepeatVector(y_length))

    # decoder
    model.add(LSTM(num_hidden, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(num_hidden, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(num_hidden, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(pitch_dimension, activation='softmax')))
    model.add(TimeDistributed(Dense(pitch_dimension, activation='softmax')))

    return model


def train_model(data_path, model_path, loss_plot):

    pianoroll = getData(data_path)
    X, Y = createTrainData(pianoroll, x_length, y_length)

    model = buildModel()
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
    earlystop = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')  # terminate training
    history = History()

    hist = model.fit(X.astype(np.bool),
                     Y.astype(np.bool),
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=[earlystop, history])

    model.save(model_path)

    img = plt.figure(dpi=75)
    plt.plot(hist.history['loss'])
    img.savefig(loss_plot, bbox_inches='tight')
