'''
Simple LSTM for Sequence Classification
'''

##########################
## Importing packages
##########################
# importing packages/function that will be useful later
from __future__ import print_function
import numpy as np
np.random.seed(1234)  # for reproducibility (manually setting random seed)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU

import matplotlib.pyplot as plt
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
import os
##########################
## Preparing data
##########################
# some parameters
vocab_size   =  15000 # number of words considered in the vocabulary
seq_length = 400  #序列长度
num_classes = 10  #类别数
# Preparing data is usually the most time-consuming part of machine learning.

base_dir = r'C:\Users\weke\Desktop\语料搜集\cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
vocab_size = len(words)
	
X_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
X_test, y_test = process_file(val_dir, word_to_id, cat_to_id, seq_length)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

## Padding input data
# Models in Keras (and elsewhere) usually take as input batches of sentences of the same length.
# Since sentences usually have different sizes, we "pad" sentences (we add a dummy "padding" token at the end of the
# sentences. The input thus has this size : (batchsize, maxseqlen) where maxseqlen is the maximum length of a sentence
# in the batch.
'''
maxlen  = 80  # cut texts after this number of words (among top vocab_size most common words)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test  = sequence.pad_sequences(X_test, maxlen=maxlen)
'''
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

##########################
## Building model
##########################

embed_dim = 32  # word embedding dimension
nhid      = 64 # number of hidden units in the LSTM
print('\nBuilding model...')

model = Sequential()
model.add(Embedding(vocab_size, embed_dim, input_length=seq_length))
model.add(LSTM(nhid, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print('Built model')

# In Keras, Torch and other deep learning framework, we create a "container" which is the Sequential() module.
# Then we add components to this contained : the lookuptable, the LSTM, the classifier etc.
# All of these components are contained in the Sequential() and are trained together.

##########################
## Define (i)   loss function
#         (ii)  optimizer
#         (iii) metrics
##########################

loss_classif     =  'binary_crossentropy'
optimizer        =  'adam' # or sgd
metrics_classif  =  ['accuracy']

# note that this is especially easy in Keras : one code line
print('\nCompiling model')
model.compile(loss=loss_classif,
              optimizer=optimizer,
              metrics=metrics_classif)
print(model.summary())
print('Compiled model')

##########################
## Train Model
##########################
validation_split =  0.2 # Held-out ("validation") data to test on.
batch_size       =  32  # size of the minibach (each batch will contain 32 sentences)
nb_epoch         =  6

# history is just an object that contains information about training.
# Look at the following line and enjoy how simple it is to train a neural network in Keras.
print('\n\nStarting training of the model\n')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split)

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(range(1,nb_epoch + 1), history.history['loss'], 'b', range(1,nb_epoch + 1), history.history['val_loss'], 'r')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(range(1,nb_epoch + 1), history.history['acc'], 'b', range(1,nb_epoch + 1), history.history['val_acc'], 'r')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

##########################
## Evaluate on test set
##########################
# evaluate model on test set (never seen during training)
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('\n\nTest loss:', score)
print('Test accuracy:', acc)
