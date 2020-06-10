# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,RNN,LSTMCell,SimpleRNNCell,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K
import keras.layers
import tensorflow as tf
from keras.callbacks import EarlyStopping

class MinimalRNNCell(keras.layers.Layer):
    """
    self defined RNNCELL, mindstorm
    """

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
        #                               initializer='uniform',
        #                               name='kernel')
        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units),
        #     initializer='uniform',
        #     name='recurrent_kernel')
        self.b = self.add_weight(shape = (self.units,),initializer='uniform')
        self.b2 = self.add_weight(shape = (self.units,),initializer='uniform')
        # self.b3 = self.add_weight(shape = (self.units,),initializer='uniform')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]

        # h = K.dot(inputs, self.kernel)
        # output = h + K.dot(prev_output, self.recurrent_kernel)+self.b
        # if K.l2_normalize(prev_output)<0.000001:
        # 	return inputs,[inputs]
        # if K.l2_normalize(inputs)<0.000001:
        # 	return prev_output,[prev_output]
        output = K.reshape(K.batch_dot(K.reshape(self.b2+prev_output,shape=(-1,side,side)),K.reshape(self.b+inputs,shape=(-1,side,side))),shape=(-1,side**2))
        print("here: ", prev_output,inputs,output)

        return output, [output]

class myRNN(RNN):
    def get_initial_state(self,inputs):
    	i = tf.broadcast_to(K.reshape(K.eye(side),shape=(1,side*side)),[K.shape(inputs)[0],side*side])
    	print("here\n\n\n",i)
    	print("")
    	return [i]
		# return tf.ones((batch_size, self.state_size))

# fix random seed for reproducibility
numpy.random.seed(7)
tf.set_random_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, embeddings_initializer = 'zeros',input_length=max_review_length))
# 4 different cells: default rnncell/GRU/LSTM/self-defined rnncell
# model.add(RNN(SimpleRNNCell(32)))
# model.add(RNN(MinimalRNNCell(32)))
model.add(GRU(32))
# model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# simple early stopping, optional
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(X_train, y_train, epochs=10, batch_size=64,validation_data = [X_test,y_test],callbacks=[es])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))