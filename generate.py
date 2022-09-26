"""
	This file loads a trained Seq2Seq LSTM model and generate music
"""

import sys
import time
import random
import glob
import numpy as np

from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam, RMSprop

from midi_parser import *

# GLOBAL PARAMETERS
x_length = 100  # sample sequence length.
y_length = 10  # output sequence length. 		Needs to be consistent with the value in train.py
iteration = 50  # number of iteration to generate new sequence. 		Final result length: y_length * itertaion


# saved_model = "./saved_params/LSTM_model.json"
# saved_weights = "./saved_params/LSTM_weights.hdf5"
# sample_folder = "./samples"
# output_folder = "./output"


def generate(model, pianoroll, tempo, resolution, output_path):

    # randomly select a sequence from the seed music
    start = np.random.randint(0, pianoroll.shape[0] - 1 - x_length - iteration)
    pattern = np.array(pianoroll[start:start + x_length])

    prediction_output = []

    # concatenate all generated sequence
    for i in range(iteration):
        # generate sequence
        prediction = model.predict(pattern.reshape(1, pattern.shape[0], -1).astype(float)).reshape(y_length, -1)
        prediction_output.append(prediction)
        pattern = np.append(pattern[y_length:, ], prediction, axis=0)  # shift sliding window on input data

    print("output shape: ", np.array(prediction_output).shape)

    # convert sequence back to piano roll
    pianoroll_output = outputPianoRoll(np.array(prediction_output), note_threshold=0.1)
    print("pianoroll shape: ", pianoroll_output.shape)

    # convert piano roll back to midi
    # scale: seqch output sequence has y_length ticks
    outputMidi(output_path, pianoroll_output, tempo, resolution, scale=int(y_length))


def generate_midi(sample_midi_path, model_path, output_path):
    model = models.load_model(model_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())

    pianoroll, tempo, resolution = parseMidi(sample_midi_path)
    generate(model, pianoroll, tempo, resolution, output_path)
