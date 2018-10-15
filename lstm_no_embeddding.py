from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LSTM, Dropout
#from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from scipy import spatial
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

## load parameters
EMBEDDING_FILE = './we_model/glove_50d_small.txt'
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 200000
MAX_SENTENCE_LENGTH = 10
## import and format text data
# format of texts going into the shredder needs to be ['sentence of words', 'another sentence of words']
with open('movie_data/X.txt') as f:
    texts = f.read().splitlines()

print(texts[0])

t = Tokenizer(MAX_VOCAB_SIZE)
def gen_2d_data(DATA_FILEPATH):
    """ Generate 2d data sentence vectors containing index per word 
    :DATA_FILEPATH: TODO
    :returns: TODO
    
    """

    with open('movie_data/X.txt') as f:
        texts = f.read().splitlines()

    t.fit_on_texts(texts)
    word_index = t.word_index
    vocab_size = len(word_index)+1    #Needed for embedding layer parameters 
    sequences = t.texts_to_sequences(texts)
    
    padded_texts = pad_sequences(sequences, MAX_SENTENCE_LENGTH, padding='post')

    x_data_index = padded_texts[0:-1,:]
    y_data_index = padded_texts[1:,:]
    
    embeddings_index = {}
    with open(EMBEDDING_FILE) as f:
        count = 0
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return()
# make keras Tokenizer object and treat text
# this paragraph will be deleted after working with gen_2d_data func
t.fit_on_texts(texts)
word_index = t.word_index
vocab_size = len(word_index)+1    #Needed for embedding layer parameters 
sequences = t.texts_to_sequences(texts)

## create padding
padded_texts = pad_sequences(sequences, MAX_SENTENCE_LENGTH, padding='post')
x_data_index = padded_texts[0:-1,:]
y_data_index = padded_texts[1:,:]
## index word vectors using glove
print('Indexing word vectors')
embeddings_index = {}
with open(EMBEDDING_FILE) as f:
    count = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


## prepare embedding matrix
print('Preparing embedding matrix')
vocab_size = min(MAX_VOCAB_SIZE, len(word_index))+1
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_matrix[0]=np.ones(EMBEDDING_DIM)
print('Null word embeddings %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print('Found %d word vectors of glove.' % len(embeddings_index))


print(embedding_matrix[x_data_index[0,0]])

## Turn y_data into word vectors
 
x_data = np.zeros((x_data_index.shape[0], x_data_index.shape[1], EMBEDDING_DIM))
y_data = np.zeros((y_data_index.shape[0], y_data_index.shape[1], EMBEDDING_DIM))
for i in range(y_data.shape[0]):
    for j in range(y_data.shape[1]):
        y_data[i,j,:] = embedding_matrix[y_data_index[i,j]]
        x_data[i,j,:] = embedding_matrix[x_data_index[i,j]]



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

## define model structure
print(EMBEDDING_DIM)
print(vocab_size)
#embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SENTENCE_LENGTH, trainable=False)
model = Sequential()
model.add(LSTM(input_shape=(MAX_SENTENCE_LENGTH, 50), return_sequences=True, activation="tanh", units=50, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal"))

#model.add(LSTM(input_shape=(MAX_SENTENCE_LENGTH, 50), return_sequences=True, activation="tanh", units=50, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", dropout=0.3))

#model.add(LSTM(input_shape=(MAX_SENTENCE_LENGTH, 50), return_sequences=True, activation="tanh", units=50, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", dropout=0.3))
model.add(LSTM(return_sequences=True, activation="tanh", units=50, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal"))
model.save('2lstm.h5')
#model.add(embedding_layer)
#model.add(LSTM(return_sequences=True, units=200, dropout=0.5))
#model.add(LSTM(return_sequences=False, units=200, dropout=0.5))
#model.add(Dropout(0.5))
#model.add(Dense((10), activation='sigmoid'))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam')
model.compile(optimizer="adam", loss='cosine_proximity', metrics=['mse', 'mae', 'mape', 'cosine'])
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3)

#print("Train...")



#early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
#result = model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_data = (x_test,y_test), metrics=['accuracy'], callbacks=[early_stopping])
#score = model.evaluate(test_X, Y_test, batch_size=batch_size, metrics=['accuracy'], verbose=1)
#print("Test Score:%d" %(score))
#lstm_layer = LSTM(
#input_list = [embedding_layer]
#model = Model(inputs=input_list, outputs=preds)
#def main():
#
#if __name__ == '__main__':
#	main()
predictions=model.predict(x_test)
print(predictions)
