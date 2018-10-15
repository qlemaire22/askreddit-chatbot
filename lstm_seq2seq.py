# source:
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html


from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
""" the above line eliminates the warning:
			"The TensorFlow library wasn't compiled to use SSE instructions,
			but these are available on your machine and could speed up CPU
			computations"
"""

from keras import backend as K

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation
import numpy as np


answers_path="data/answers995.txt"
questions_path="data/questions995.txt"
embedding_path="data/glove.6B.50d.txt"

embedding_dimensions=50

epochs=1;

batch_size=32;

latent_dim=100


def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


word_count=0;
def prepare_data():
	print('hey')
	vocab={};
	specific_vocab={};
	with open(embedding_path,'r', encoding='utf-8') as f:
		num_lines = sum(1 for line in f) # careful here, the number of lines may decrease if we remove some datatpoints
	with open(embedding_path,'r', encoding='utf-8') as f:
		word_count=0
		#load str to vector dic
		gloveMatrix=np.zeros((num_lines,embedding_dimensions),dtype=float)
		gloveList=[]
		for line in f:
			if line is not "\n":

				cells=line.strip("\n").split(" ");
				vocab[cells[0]] = np.fromiter([x for x in cells[1:]],dtype=float).reshape((50,))
				gloveList.append(cells[0]);
				gloveMatrix[word_count,:]=np.fromiter([x for x in cells[1:]],dtype=float).reshape(50,);
				word_count=word_count+1;

	X=[]
	Y=[]

	n_unseen_word_occurences=0;
	with open(questions_path,'r', encoding='utf-8') as q, \
            open(answers_path,'r', encoding='utf-8') as a:

		qlines=q.readlines(); alines=a.readlines()
		#turn sentences into word vector sequences
		for ql,al in zip(qlines,alines):
			qcells=ql.strip().split(" "); acells=al.strip().split(" ")
			qvector_seq=[]; avector_seq=[]
			for qword in qcells[1:]:
				if qword in vocab:
					qvector_seq.append(vocab[qword]);
					if qword not in specific_vocab:
						specific_vocab[qword] = vocab[qword];
				else:
					n_unseen_word_occurences+=1;
			for aword in acells[1:]:
				if aword in vocab:
					avector_seq.append(vocab[aword]);
					if aword not in specific_vocab:
						specific_vocab[aword] = vocab[aword];
				else:
					n_unseen_word_occurences+=1;

					#vector_seq.append(vocab["s"]);#replaces unseen words
			if len(qvector_seq) and len(avector_seq) > 0:
				X.append(qvector_seq); Y.append(avector_seq);

	print("they were {} occurences of unseen words".format(n_unseen_word_occurences));

	return X,Y,gloveMatrix,gloveList, specific_vocab;


X,Y,gloveMatrix,gloveList, specific_glove = prepare_data();
print(len(X),len(Y))

start_word=np.ones((50,));
stop_word=-np.ones((50,));

gloveMatrix=np.vstack((gloveMatrix,start_word,stop_word))
gloveList.append('start_word')
gloveList.append('stop_word')
# add start and stop special words to every sequence
for y in Y: y.insert(0, start_word); y.append(stop_word)


max_encoder_seq_length = max([len(x) for x in X])
max_decoder_seq_length = max([len(y) for y in Y])

print('Number of samples:', len(X))
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

encoder_input_data = np.zeros( # change to empty for speed purposes?
	(len(X), max_encoder_seq_length, embedding_dimensions),
	dtype='float32')
decoder_input_data = np.zeros(
	(len(X), max_decoder_seq_length, embedding_dimensions),
	dtype='float32')
decoder_target_data = np.zeros(
	(len(X), max_decoder_seq_length, embedding_dimensions),
	dtype='float32')


for i, (input_text, target_text) in enumerate(zip(X, Y)):
	for t, word in enumerate(input_text):
		encoder_input_data[i, t] = word;
	for t, word in enumerate(target_text):
		# decoder_target_data is ahead of decoder_input_data by one timestep
		decoder_input_data[i, t] = word;
		if t > 0:
			# decoder_target_data will be ahead by one timestep
			# and will not include the start character.
			decoder_target_data[i, t - 1] = word;

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, embedding_dimensions))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, embedding_dimensions))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
									 initial_state=encoder_states)
decoder_dense = Dense(embedding_dimensions, activation='linear') #fix activation (leaky relu)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='cosine_proximity' ,metrics=["mae","acc"])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
		  batch_size=batch_size,
		  epochs=epochs,
		  validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
	decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
	[decoder_inputs] + decoder_states_inputs,
	[decoder_outputs] + decoder_states)

'''
# Reverse-lookup token index to decode sequences back to
# something readable.


reverse_input_char_index = dict(
	(i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
	(i, char) for char, i in target_token_index.items())
'''


def decode_sequence(input_seq,embedding_matrix):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, embedding_dimensions))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0] = start_word

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_vector, h, c = decoder_model.predict(
			[target_seq] + states_value)

		# Sample a token
		match_vector=embedding_matrix@output_vector[0][0]
		new_word=gloveList[np.argmax(match_vector)]
		decoded_sentence+=new_word

		'sampled_token_index = np.argmax(output_tokens[0, -1, :])'
		'sampled_char = reverse_target_char_index[sampled_token_index]'

		# Exit condition: either hit max length
		# or find stop character.
		if (new_word == 'stop_word' or
		   len(decoded_sentence) > max_decoder_seq_length):
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, embedding_dimensions))
		target_seq[0, 0] = output_vector[0][0];   ####THINK!!!

		# Update states
		states_value = [h, c]

	return decoded_sentence


for seq_index in range(100):
	# Take one sequence (part of the training set)
	# for trying out decoding.
	input_seq = encoder_input_data[seq_index: seq_index + 1]
	decoded_sentence = decode_sequence(input_seq,gloveMatrix)
	print('-')
	#print('Input sentence:', input_texts[seq_index])
print('Decoded sentence:', decoded_sentence)
