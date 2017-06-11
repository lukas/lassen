# source :
# http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model

import click

CACHE_FILENAME = 'wonderdland_preprocess_cache.pickle'



@click.group()
def cli():
    pass

@cli.command()
def generate_text():
	with open(CACHE_FILENAME, 'rb') as input:
		cached_data = pickle.load(input)
		char_to_int = cached_data['char_to_int']
		int_to_char = cached_data['int_to_char']
		X = cached_data['X']
		y = cached_data['y']
		print("Read cache file %s." % input.name)

	# seq_length = 100
	# dataX, dataY = [], []
	# for i in range(len(raw_text) - seq_length):
	# 	seq_in = raw_text[i:i + seq_length]
	# 	seq_out = raw_text[i + seq_length]
	# 	dataX.append([char_to_int[char] for char in seq_in])
	# 	dataY.append(char_to_int[seq_out])

	start = np.random.randint(0, len(X)-1)
	pattern = [int(x) for x in X[start] * len(char_to_int)]
	#pattern = pattern[:10]
	#print("".join(int_to_char[i] for i in pattern))

	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.load_weights('weights-improvement-00-2.9801.hdf5')

	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	# generate characters
	for i in range(10000):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(len(int_to_char))
		prediction = model.predict(x, verbose=0)
		#print(prediction[0])
		index = np.random.choice(range(len(prediction[0])), p = prediction[0] ** 2 / sum(prediction[0] ** 2))
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		sys.stdout.flush()
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\nDone.")

@cli.command()
def preprocess_text():
	# load ascii text and covert to lowercase
	filename = "wonderland.txt"
	raw_text = open(filename).read()
	raw_text = raw_text.lower()

	# create mapping of unique chars to integers
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i,c) for (c,i) in char_to_int.items())

	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX, dataY = [], []
	for i in range(len(raw_text) - seq_length):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)

	# reshape X to be [samples, time steps, features]
	X = np.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(len(chars))
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)

	print(len(char_to_int))
	print(len(int_to_char))
	print(X.shape)
	print(y.shape)

	# save the preprocessed text
	cached_data = {
		'char_to_int': char_to_int,
		'int_to_char': int_to_char,
		'X': X,
		'y': y
	}
	with open(CACHE_FILENAME, 'wb') as output:
		pickle.dump(cached_data, output)
		print("Wrote cache to %s." % output.name)

@cli.command()
def learn_model():
	with open(CACHE_FILENAME, 'rb') as input:
		cached_data = pickle.load(input)
		char_to_int = cached_data['char_to_int']
		int_to_char = cached_data['int_to_char']
		X = cached_data['X']
		y = cached_data['y']
		print("Read cache file %s." % input.name)

	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

if __name__ == "__main__":
    cli()
