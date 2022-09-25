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


def generate(model, input_data, tempo, resolution, output_path):
    '''
		generate new music and save to a midi file

		params:
				input_data: seed music pianoroll for music generation
				tempo: tempo value parsed from the seed music
				resolution: resolution value parsed from the seed music
	'''

    # output_path = os.path.join(output_path, "generated_%s.midi" % (time.strftime("%Y%m%d_%H_%M")))

    # randomly select a sequence from the seed music
    start = np.random.randint(0, input_data.shape[0] - 1 - x_length - iteration)
    pattern = np.array(input_data[start:start + x_length])

    prediction_output = []

    # concatenate all generated sequence
    for i in range(iteration):
        prediction = model.predict(pattern.reshape(1, pattern.shape[0], -1).astype(float)).reshape(y_length,
                                                                                                   -1)  # generate sequence
        prediction_output.append(prediction)
        pattern = np.append(pattern[y_length:, ], prediction, axis=0)  # shift sliding window on input data

    print("output shape: ", np.array(prediction_output).shape)

    # convert sequence back to piano roll
    pianoroll = outputPianoRoll(np.array(prediction_output), note_threshold=0.1)
    print("pianoroll shape: ", pianoroll.shape)

    # convert piano roll back to midi
    # scale: seqch output sequence has y_length ticks
    outputMidi(output_path, pianoroll, tempo, resolution, scale=int(y_length))


def generate_midi(data_path, model_path, output_path):

    model = models.load_model(model_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop())

    # randomly select a file from sample folder
    midi_files = [file for file in os.listdir(data_path) if file.endswith(".midi") or file.endswith(".mid")]
    input_data, tempo, resolution = parseMidi(
        os.path.join(data_path,
                     midi_files[random.randint(0, len(midi_files) - 1)]))

    generate(model, input_data, tempo, resolution, output_path)
