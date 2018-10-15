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
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Masking, Embedding, BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import argparse

from sklearn.metrics.pairwise import pairwise_distances as sklearn_cos

def load_data(args):
	vec_path = args.glove_path + '/' + 'glove.6B.' + str(args.vector_dim) + 'd.txt'
	cache_path = 'data/cache/dataset_' + str(args.vector_dim) + 'd.npz'
	vec_list = []
	wd_to_id = {}
	id_to_wd = {}
	word_dict = {}
	max_len_que = args.max_q
	max_len_ans = args.max_a
	# load word embeddings
	print('Loading GloVe vectors...')
	with open(vec_path, 'r', encoding='utf-8') as f:
		for i, line in enumerate(f):
			entry = line.split()
			wd_to_id[entry[0]] = i
			id_to_wd[i] = entry[0]
			vec_list.append(np.array([float(j) for j in entry[1:]]))
			word_dict[entry[0]] = vec_list[i]
			if i == args.num_words - 1:
				break
	print('Loaded {} {}-dimensional GloVe vectors.'.format(i+1, args.vector_dim))
	embed_mat = np.array(vec_list)

	if not os.path.isfile(cache_path):
		num_unk_questions = 0; num_unk_answers = 0
		num_known_questions = 0; num_known_answers = 0
		qa_dict = {}
		X, Y = [], []
		# load dataset
		with open(args.qa_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()
			offset = 1; sep = 3
			#turn sentences into word vector sequences
			for line_que, line_ans in [lines[offset+x:x+sep] for x in range(0, len(lines), offset + sep)]:
				words_que = line_que.strip().split(' ')
				words_ans = line_ans.strip().split(' ')
				seq_que, seq_ans = [], []
				# if len(words_ans) > 120:
				# 	print('\n\n',' '.join(words_ans))
				for w_q in words_que:
					if w_q in word_dict:
						num_known_questions += 1
						seq_que.append(word_dict[w_q])
						if w_q not in qa_dict:
							qa_dict[w_q] = word_dict[w_q] # great impact on speed
					else:
						# print(w_q,end=' ')
						num_unk_questions += 1
				for w_a in words_ans:
					if w_a in word_dict:
						num_known_answers += 1
						seq_ans.append(word_dict[w_a])
						if w_a not in qa_dict:
							qa_dict[w_a] = word_dict[w_a] # great impact on speed
					else:
						num_unk_answers += 1
						#vector_seq.append(vocab["s"]);#replaces unseen words
				if len(seq_que) > 0 and len(seq_ans) > 0 and \
						len(seq_que) <= max_len_que and len(seq_ans) <= max_len_ans:
					X.append(seq_que); Y.append(seq_ans);
		print(num_unk_questions,'unknown words in the questions. ')
		print(num_unk_answers,'unknown words in the answers.')
		np.savez_compressed(cache_path, X=X, Y=Y, qa_dict=qa_dict)
		print('Saved compressed dataset to cache.')
	else:
		dataset = np.load(cache_path)
		X, Y, qa_dict = dataset['X'].tolist(), dataset['Y'].tolist(), dataset['qa_dict'].item()
		print('Loaded compressed dataset from cache.')

	return X, Y, embed_mat, wd_to_id, id_to_wd, qa_dict

def find_nearest(vec, id_to_wd, embed_mat, num_results, method='cosine'):
	res = []
	if method == 'cosine':
		similarity = np.dot(embed_mat, vec)
		# squared magnitude of preference vectors (number of occurrences)
		square_mag = np.diag(similarity)
		# inverse squared magnitude
		inv_square_mag = 1 / square_mag
		# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
		inv_square_mag[np.isinf(inv_square_mag)] = 0
		# inverse of the magnitude
		inv_mag = np.sqrt(inv_square_mag)
		# cosine similarity (elementwise multiply by inverse magnitudes)
		cosine = similarity * inv_mag
		cosine = cosine.T * inv_mag
	elif method == 'sk_cos':
		cosine = sklearn_cos(embed_mat, vec.T, metric='cosine', n_jobs=-1)
	else:
		raise Exception('{} is not an excepted method parameter'.format(method))

	for _ in range(num_results):
		best = np.argmax(cosine)
		res.append(id_to_wd[best])
		# delete row corresponding to the best candidate
		cosine = np.delete(cosine, best, axis=0 if method=='sk_cos' else 1)

	return res

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vector_dim', '-d',
						type=int,
						choices=[50, 100, 200, 300],
						default=50,
						help='What vector GloVe vector dimension to use '
							 '(default: 50).')
	parser.add_argument('--h1_dim', '-l',
						type=int,
						default=128,
						help='What latent LSTM dimension to use '
							 '(default: 128).')
	parser.add_argument('--batch_size', '-b',
						type=int,
						default=64,
						help='What batch size to use '
							 '(default: 64).')
	parser.add_argument('--n_epochs', '-e',
						type=int,
						default=50,
						help='Number of epochs for training '
							 '(default: 50).')
	parser.add_argument('--num_words', '-n',
						type=int,
						default=400000,
						help='The number of lines to read from the GloVe '
							 'vector file (default: ALL).')
	parser.add_argument('--max_q', '-mq',
						type=int,
						default=25,
						help='Maximum length of question sequences '
							 '(default: 25).')
	parser.add_argument('--max_a', '-ma',
						type=int,
						default=50,
						help='Maximum length of answers sequences '
							 '(default: 50).')
	parser.add_argument('--glove_path', '-g',
		                default='data/glove',
		                help='GloVe vector file path (default: data/glove)')
	parser.add_argument('--qa_path', '-qa',
		                default='data/askredditData.txt',
		                help='Q&A vector file path (default: data/askredditData.txt)')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	embedding_dim = args.vector_dim
	latent_dim = args.h1_dim
	batch_size = args.batch_size
	epochs = args.n_epochs
	que_mark_idx = 189
	epochs = 10
	batch_size = 128
	latent_dim = 128

	model_path = 'data/cache/model.h5'
	if not os.path.isfile(model_path):
		X, Y, embedding_matrix, word_to_idx, idx_to_word, qa_lexicon = load_data(args)
		# print(idx_to_word[que_mark_idx])
		# cand = find_nearest(np.array([qa_lexicon[idx_to_word[que_mark_idx]]]).T, idx_to_word, embedding_matrix, 1, method='sci_cos')
		# print(cand)
		size_lexicon = len(embedding_matrix)
		# 189 is the index of the question mark char
		word_to_idx['START_ANS'] = size_lexicon; word_to_idx['STOP_ANS'] = size_lexicon+1
		idx_to_word[size_lexicon] = 'START_ANS'; idx_to_word[size_lexicon+1] = 'STOP_ANS'
		start_ans = 0.01*np.ones((embedding_dim,)); stop_ans = 0.1*np.ones((embedding_dim,))
		embedding_matrix = np.vstack((embedding_matrix, start_ans, stop_ans))
		# add start and stop special words to every sequence
		# for x in X: x.append(qa_lexicon['?'])
		# for y in Y: y.append(qa_lexicon['.'])
		for y in Y: y.append(stop_ans)
		# print(len(X), len(Y), len(X[0]), len(Y[0]), len(lexicon), len(qa_lexicon), embedding_matrix.shape)
		# define maximum lengths for the encoder and decoder LSTMs
		max_encoder_seq_length = max([len(x) for x in X])
		max_decoder_seq_length = max([len(y) for y in Y])
		print('Max sequence length for questions:', max_encoder_seq_length)
		print('Max sequence length for answers:', max_decoder_seq_length)
		# initialize data containers
		encoder_input_data = np.zeros((len(X), max_encoder_seq_length, embedding_dim), dtype='float32')
		decoder_input_data = np.zeros((len(X), max_decoder_seq_length, embedding_dim), dtype='float32')
		decoder_target_data = np.zeros((len(X), max_decoder_seq_length, embedding_dim), dtype='float32')
		# fill with values
		for i, (input_text, target_text) in enumerate(zip(X, Y)):
			for t_inp, wordvec in enumerate(input_text):
				encoder_input_data[i,t_inp] = wordvec
			for t_tar, wordvec in enumerate(target_text):
				# decoder_target_data is ahead of decoder_input_data by one timestep
				decoder_target_data[i,t_tar] = wordvec
				if t_tar == 0:
					decoder_input_data[i,t_tar] = start_ans
				else:
					decoder_input_data[i,t_tar] = decoder_target_data[i,t_tar-1]

		print(np.count_nonzero(encoder_input_data==0)//embedding_dim,'zero-words in encoder_input_data.')
		print(np.count_nonzero(decoder_input_data==0)//embedding_dim,'zero-words in decoder_input_data.')
		print(np.count_nonzero(decoder_target_data==0)//embedding_dim,'zero-words in decoder_target_data.')

		# MODEL DEFINITION
		print('Build model...')
		# Define an input sequence and process it.
		encoder_inputs = Input(shape=(None, embedding_dim))
		encoder_masked_inputs = Masking(mask_value=0., input_shape=(None, embedding_dim))(encoder_inputs)
		encoder = LSTM(latent_dim, return_sequences=False, return_state=True, activation='tanh')
		encoder_outputs, state_h, state_c = encoder(encoder_masked_inputs)
		# We discard `encoder_outputs` and only keep the states.
		encoder_states = [state_h, state_c]
		# Set up the decoder, using `encoder_states` as initial state.
		decoder_inputs = Input(shape=(None, embedding_dim))
		decoder_masked_inputs = Masking(mask_value=0., input_shape=(None, embedding_dim))(decoder_inputs)
		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the
		# return states in the training model, but we will use them in inference.
		decoder = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh')
		decoder_outputs, _, _ = decoder(decoder_masked_inputs,
											 initial_state=encoder_states)
		decoder_dense = Dense(embedding_dim, activation='tanh') #fix activation (leaky relu)
		decoder_outputs = decoder_dense(decoder_outputs)
		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
		model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, verbose=0, mode='auto')
		# Run training
		model.compile(optimizer='adagrad', loss='cosine_proximity' ,metrics=['accuracy'])
		model.summary()
		print('Train...')
		model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
				  batch_size=batch_size,
				  epochs=epochs,
				  validation_split=0.2,
				  callbacks=[early_stop])
		# Save model
		# model.save(model_path)
	else:
		model = load_model(model_path)

	# INFERENCE MODEL>>>>>>>>>>>>IS IT POSSIBLE TO MASK HERE TOO????
	encoder_model = Model(encoder_inputs, encoder_states)
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder(
	    decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model(
	    [decoder_inputs] + decoder_states_inputs,
	    [decoder_outputs] + decoder_states)

	Npred = 10
	num_candidates = 5
	results = []
	for question, seq_len in zip(encoder_input_data[:Npred], [len(x) for x in X[:Npred]]):
	    qwords = []
	    for qw in question[:seq_len]:
			# vector shape must be (embed_dim, 1)
	        qwords.append(find_nearest(np.array([qw]).T, idx_to_word, embedding_matrix, 1)[0])
	    # Encode the input as state vectors.
	    states = encoder_model.predict(np.array([question[:seq_len]]))
	    # Generate empty target sequence of length 1.
	    target_seq = np.zeros((1, 1, embedding_dim))
	    # Populate the first word of target sequence with the start_ans word.
	    target_seq[0, 0, :] = start_ans
	    # Sampling loop for a batch of sequences
	    # (to simplify, here we assume a batch of size 1).
	    stop_condition = False
	    ans_len = 0
	    while not stop_condition:
	        ans_vec, h, c = decoder_model.predict([target_seq] + states)
	        if ans_len == 0:
	        	print()
	        	print('\n'.join([str(entry) for entry in ans_vec[0][0]]))
	        	print()
	        # Sample a word
	        candidates = find_nearest(ans_vec[0].T, idx_to_word, embedding_matrix, num_candidates)
	        results.append(candidates)
	        ans_len += 1
	        # Exit condition: either hit max length
	        # or find stop character.
	        if ('.' in candidates or 'STOP_ANS' in candidates or
	           ans_len > max_decoder_seq_length):
	            stop_condition = True
	        # Update the target sequence (of length 1).
	        target_seq = np.zeros((1, 1, embedding_dim))
	        target_seq[0, 0, :] = embedding_matrix[word_to_idx[candidates[0]]]
	        # target_seq[0, 0, :] = embedding_matrix[word_to_idx[np.random.choice(candidates,replace=False)]]
	        # Update states
	        states = [h, c]
	    print('QUESTION:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	    print(' '.join(qwords))
	    print('ANSWER:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	    print(' '.join([res[0] for res in results]))
	    # print(' '.join([np.random.choice(results,replace=False) for _ in results]))
	# print(results)
	from collections import Counter
	print(Counter(x for xs in results for x in xs))
