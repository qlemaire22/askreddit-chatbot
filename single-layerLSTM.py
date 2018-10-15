import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
""" the above line eliminates the warning:
			"The TensorFlow library wasn't compiled to use SSE instructions,
			but these are available on your machine and could speed up CPU
			computations"
"""

import keras.backend as K
from keras.layers import Input, Dense, Embedding, merge, Flatten, SimpleRNN, LSTM
from keras.models import Model, Sequential

def load_dataset(word_vec_vocab,examples_file,targets_file,dim,qminlen=150,amaxlen=150):

	vocab={};
	set_specific_vocab={};
	with open(word_vec_vocab,'r', encoding='utf-8') as f:

		#load str to vector dic
		for line in f:
			if line is not "\n":
				cells=line.strip("\n").split(" ");
				#print(cells[1:301])
				vocab[cells[0]] = np.fromiter([x for x in cells[1:dim+1]],dtype=float).reshape((dim,1))

	n_unseen_word_occurences = 0
	n_known_word_occurences = 0
	n_number_of_zeroes = 0
	X, Y = [], []
	qsentences, asentences = [], []
	with open(examples_file, 'r', encoding='utf-8') as q, open(targets_file, 'r', encoding='utf-8') as a:
		questions = q.readlines()
		answers = a.readlines()
		#turn sentences into word vector sequences
		print("total number of questions: {}".format(len(questions)))
		print("total number of answers: {}".format(len(answers)))
		no_punct = ".replace('[',' ').replace(']',' ').replace('(',' ') \
						.replace(',',' ').replace(')',' ').replace('/',' ') \
						.replace('.',' ').replace('-',' ')"
		for que, ans in filter(lambda qa:
			# conditions to filter out undesirable Q&A pairs
			len(qa[0]) > qminlen and len(qa[1]) < amaxlen and qa[0][-2] == '?',
				zip(questions,answers)):

			qcells = eval("que"+no_punct).lower().strip('\n').split(' ')
			acells = eval("ans"+no_punct).lower().strip('\n').split(' ')
			# strip the question mark from the last word
			qcells[-1] = qcells[-1][:-1]
			# add it as the last element of the sequence
			qcells = qcells + ['?']
			qlabelTmp, alabelTmp = qcells[0], acells[0]
			if qlabelTmp == '0' or alabelTmp == '0':
					n_number_of_zeroes += 1
			qwords, awords = qcells[1:], acells[1:]
			qvector_seq, avector_seq = [], []

			for qw in qwords:
				if qw in vocab:
					n_known_word_occurences += 1
					qvector_seq.append(vocab[qw].flatten()) # added flatten
					if qw not in set_specific_vocab:
						set_specific_vocab[qw] = vocab[qw]
				else: n_unseen_word_occurences += 1
			for aw in awords:
				if aw in vocab:
					n_known_word_occurences += 1
					avector_seq.append(vocab[aw].flatten()) # added flatten
					if aw not in set_specific_vocab:
						set_specific_vocab[aw] = vocab[aw]
				else: n_unseen_word_occurences += 1


			if len(qvector_seq) and len(avector_seq) > 1:
				# random.shuffle(vector_seq) # for suffling words in every sentence
			 	# append each sentence to the training set
				qsentences.append(qwords + awords[:-1])
				X.append(np.asarray([qvector_seq + avector_seq[:-1]]))
				idxs = np.array(range(len(qvector_seq)-1))
				np.random.shuffle(idxs)
				total_len = len(qvector_seq) + len(avector_seq) - 1
				while len(avector_seq) < total_len:
					idx = np.random.choice(idxs)
					try:
						awords.insert(0, awords[idx])
						avector_seq.insert(0, avector_seq[idx])
					except IndexError:
						pass
				asentences.append(awords)
				Y.append(np.asarray([avector_seq]))

	print("there were {} occurences of unseen words".format(n_unseen_word_occurences));
	print("there were {} occurences of known words".format(n_known_word_occurences));
	print("zeros = {}".format(n_number_of_zeroes));

	return X,Y, qsentences,asentences, vocab,set_specific_vocab

# incremental average function for efficient loss and acc updating
# https://math.stackexchange.com/questions/106700/incremental-averageing
def inc_avg(last_avg, new_elem, n):
    return last_avg + (new_elem-last_avg)/n
# at the end of each epoch we need to compute a weighted avg for the remaining datapoints
def weighted_avg(last_avg, new_avg, w1, w2):
	return (w1*last_avg + w2*new_avg)/(w1+w2)
# custom accuracy metric so that only the last timestep of y_pred for each seq is considered
def custom_acc_last(y_true, y_pred):
    return K.expand_dims(K.mean(
			K.equal(K.expand_dims(y_true[:,-1,:],axis=-2),
			K.round(K.expand_dims(y_pred[:,-1,:],axis=-2))),
			axis=-1),axis=-1)

# https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
def cos_sim(vA,vB):
	# base similarity matrix (all dot products)
	# replace this with A.dot(A.T).toarray() for sparse representation
	similarity = np.dot(vA, vB)
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
	return np.argmax(cosine)

"""
- modify qminlen and amaxlen to choose the restrictions on the questions and
  answers lengths


"""
# >>>START>>>
# load dataset
qminlen = 10
amaxlen = 100
vocab_filename = 'data/glove.6B.50d.txt'
questions_filename = 'data/questions995.txt'
answers_filename = 'data/answers995.txt'
X,Y, questions,answers, vocab,set_specific_vocab = \
	load_dataset(vocab_filename,questions_filename,answers_filename,50,qminlen,amaxlen)
# create embedding matrix
vocab_w, vocab_v = [], []
for word, vec in set_specific_vocab.items():
	vocab_w.append(word)
	vocab_v.append(vec)
vocab_m = np.hstack(vocab_v).T

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PRINTS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# print(X[0].shape, Y[0].shape)
# print(vocab_m.shape)
# print(answers[77])

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# dimensions
N = len(X)

size_embed = 50 # size of each feature word vector
size_hidden1 = 50 # size of the hidden state vector of the 1st LSTM layer
# hyperparameters
nepochs = 1
batch_size = 1
# split dataset
test2train_ratio = 0.2
val2train_ratio = 0.2
size_test = int(test2train_ratio*N)
size_val = int(val2train_ratio*(N-size_test))
size_train = N - size_test - size_val
X_test, Y_test = X[-size_test:], Y[-size_test:]

# validation set currently not used
X_val, Y_val = X[size_train:-size_test], Y[size_train:-size_test]
X_train, Y_train = X[:size_train], Y[:size_train]
# >>>Important info: https://github.com/keras-team/keras/issues/85
	# @Kevinpsk You cannot batch sequences of different length together.
	# @Binteislam (batch_input_shape, time_length, input_dim) = (1, None, input_dim)
print('Build model...')
model = Sequential()
lstm1 = LSTM(size_hidden1, return_sequences=True, input_shape=(None, size_embed), activation='tanh')
# sigmoid1 = Dense(1, activation='sigmoid')
model.add(lstm1)
# lstm1_in = Input(shape=(None, size_embed)) # unknown timespan, fixed feature size
# lstm1_out = K.function(inputs=[lstm1_in], outputs=[lstm1(lstm1_in)])
# model.add(sigmoid1) # many to one
# sigmoid1_in = Input(shape=(None, size_hidden1))
# sigmoid1_out = K.function(inputs=[sigmoid1_in], outputs=[sigmoid1(sigmoid1_in)])
print('Compile model...')
# we have to try using different optimizers and different optimizer configs
model.compile(loss='cosine_proximity',# 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model.fit does not accept a list as input (which allows for different sentence lengths)
# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           epochs=nepochs,
#           validation_data=(X_val, Y_val))
print('Train...')
train_metrics = np.empty((size_train,2)) # loss and accuracy for each input batch
loss, acc = 0, 0
step_len = 100 # every step_len iterations, metrics will be printed
step_max_train = size_train//step_len+1 if (size_train%step_len)!=0 else size_train//step_len
step_max_test = size_test//step_len+1 if (size_test%step_len)!=0 else size_test//step_len
pred_author_train = []
pred_author_train2 = []
for e in range(nepochs):
	for i, (x_train,y_train) in enumerate(zip(X_train, Y_train)):
		# considering all predictions for each sequence
		train_metrics[i] = model.train_on_batch(x_train, y_train)
		# consider only the last prediction for each sequence
		# pred_author_train.append(sigmoid1_out(lstm1_out([x_train]))[0][0,-1,0])
		# pred_author_train2.append(model.predict(x_train)[0,-1,0])
		# maybe perform the previous computations outside the training loop? (see evaluation loop)
		if (i+1) % step_len == 0:
			nsteps = (i+1)//step_len
			loss = inc_avg(loss, np.mean(train_metrics[(nsteps-1)*step_len:i+1,0]), nsteps+e*step_max_train)
			acc = inc_avg(acc, np.mean(train_metrics[(nsteps-1)*step_len:i+1,1]), nsteps+e*step_max_train)
			print('iteration {2: <5} > loss: {0} - acc: {1}'.format(loss,acc,i+1))
	if (size_train % step_len) != 0: # all batch means computed, update means with remaining metrics
		loss = weighted_avg(loss, np.mean(train_metrics[(step_max_train-1)*step_len:,0]),
							(step_max_train-1)*step_len+e*size_train, size_train-(step_max_train-1)*step_len)
		acc = weighted_avg(acc, np.mean(train_metrics[(step_max_train-1)*step_len:,1]),
							(step_max_train-1)*step_len+e*size_train, size_train-(step_max_train-1)*step_len)
	print('Epoch {0}/{1} > loss: {2} - acc: {3} - val_loss: {4} - val_acc: {5}'.format(e+1,nepochs,loss,acc,0,0))

# print('Training acc computed by directly averaging over all saved predictions >>>',
	# sum([round(auth)==y[0][0][0] for auth, y in zip(pred_author_train, Y_train*nepochs)])/size_train*nepochs)
# print('The same but with the predictions computed in a different way >>>',
# 	sum([round(auth)==y[0][0][0] for auth, y in zip(pred_author_train2, Y_train*nepochs)])/size_train*nepochs)
print('Evaluate...')
# reinitialize metrics
loss, acc = 0, 0
test_metrics = np.empty((size_test,2))
words_v = []
pred_author_test = []
pred_author_test2 = []
for i, (x_test,y_test) in enumerate(zip(X_test, Y_test)):
	# considering all predictions for each sequence
	test_metrics[i] = model.test_on_batch(x_test, y_test)
	words_v.append(model.predict(x_test))
	# consider only the last prediction for each sequence
	# pred_author_test.append(sigmoid1_out(lstm1_out([x_test]))[0][0,-1,0])
	# pred_author_test2.append(model.predict(x_test)[0,-1,0])
	if (i+1) % step_len == 0:
		nsteps = (i+1)//step_len
		loss = inc_avg(loss, np.mean(test_metrics[(nsteps-1)*step_len:i+1,0]), nsteps)
		acc = inc_avg(acc, np.mean(test_metrics[(nsteps-1)*step_len:i+1,1]), nsteps)
		print('iteration {2: <5} > loss: {0} - acc: {1}'.format(loss,acc,i+1))
if (size_test % step_len) != 0: # all batch means computed, update means with remaining metrics
	loss = weighted_avg(loss, np.mean(test_metrics[(step_max_test-1)*step_len:,0]),
						(step_max_test-1)*step_len, size_test-(step_max_test-1)*step_len)
	acc = weighted_avg(acc, np.mean(test_metrics[(step_max_test-1)*step_len:,1]),
						(step_max_test-1)*step_len, size_test-(step_max_test-1)*step_len)
print('Evaluation results > loss: {0} - acc: {1}'.format(loss,acc))
# print('Evaluation acc computed by directly averaging over all saved predictions >>>',
	# sum([round(auth)==y[0][0][0] for auth, y in zip(pred_author_test, Y_test)])/size_test)
# print('The same but with the predictions computed in a different way >>>',
# 	sum([round(auth)==y[0][0][0] for auth, y in zip(pred_author_test2, Y_test)])/size_test)
# model.evaluate does not accept a list as input (which allows for different sentence lengths)
# score, acc = model.evaluate(X_test, Y_test,
#                             batch_size=batch_size)


from collections import defaultdict
d = defaultdict(int)
print('\n\nNumber of questions in X_test:',len(X_test))
for i, (x_test,y_test) in enumerate(zip(X_test, Y_test)):
	pred = model.predict(x_test)[0] # am I getting the predictions for all timesteps here?
	for p in pred:
		index = cos_sim(vocab_m,np.array([p]).T) # not sure if cos_sim implementation is correct
		d[vocab_w[index]] += 1

for k,v in d.items():
	print(k,v)
