#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import string
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, Dropout, Activation, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras import optimizers
from keras import regularizers
from keras.engine import InputSpec, Layer
import random
from attention import *
import os
import pickle
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report
from keras import backend as K


np.random.seed(22)                              # to reproduce results
random.seed(22)


# PARAMETERS 

data_dir = ''                                   # path to dataset directory
training_file = 'training_data.txt'                           # traning file
max_len = 15                                    # max words in a query
max_words = 4000                                # max number of words in vocab, max_vocab_size function dynamically finds the vacab size based on input data
hidden_size=128                                 # LSTM units
embed_size=128                                  # embedding dimension
lr=0.001
decay=1e-7
test_size=.15                                   # for sklearn train test split
load_prev_model = True                         
model_pretrained='model_files/model.h5'         # model path to save model or load pretrained model
tokenizer_path='model_files/tokenizer.pickle'   # tokenizer path to save tokenizer
batch_size=128
epochs=10
validation_split=0.2







def remove_punc(sentence):
    
    punct_set= set(string.punctuation)#Saving punctuation into a set
    s = ''.join(ch for ch in sentence if ch not in punct_set)
    return s



def indices_to_one_hot(x, n_classes):
    return np.eye(n_classes)[x]

def preprocess(filename):
    data = open(filename)
    with data as f:
        lines = [line.rstrip() for line in f]
        query=[li.split('\t')[0].strip() for li in lines]
        query=[remove_punc(li) for li in query]
        
        
        label=[li.split('\t')[1].strip() for li in lines]
        label=[int(lab) for lab in label]
        label=indices_to_one_hot(label,2)

        
    return query,label

def max_vocab_size(X_train):
    query_word_list=''
    for query in X_train:
        query_word_list=str(query)+' '+query_word_list

    c=Counter(query_word_list.split())
    word_list = [word for word,count in c.items() if count > 1]
    max_words=len(word_list)
    return max_words

 

X,Y=preprocess(data_dir+training_file)



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,stratify=Y,shuffle=True)
# max_words=max_vocab_size(X_train)

tokenizer = Tokenizer(num_words=max_words+1,oov_token='UNK')
tokenizer.fit_on_texts(X_train)
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

sequences = tokenizer.texts_to_sequences(X_train)
X_train_sequences= sequence.pad_sequences(sequences,maxlen=max_len)


gru_dropout=.00002
def build_model():
    input_matrix = Input(name='inputs',shape=[max_len])
    embed_layer = Embedding(max_words+1,embed_size,trainable=True,mask_zero=True)(input_matrix)
    lstm_output =LSTM(hidden_size,return_sequences=True,
                          kernel_regularizer=regularizers.l2(gru_dropout),
                          recurrent_regularizer=regularizers.l2(gru_dropout),recurrent_activation='tanh',use_bias=True,
    kernel_initializer="normal",
    recurrent_initializer="normal")(embed_layer)
    context_attention = Attention()(lstm_output)
    predictions = Dense(Y_train.shape[1], activation="softmax")(context_attention[0])
    model = Model(inputs=input_matrix, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(lr, decay=decay),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model


model=build_model()

if load_prev_model:
    model=load_model(model_pretrained, custom_objects={'Attention': Attention})

model.fit(X_train_sequences,Y_train,batch_size=batch_size,epochs=10,
          validation_split=validation_split,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
model.save('model_files/model.h5')
print(model.summary())




##testing
sequences = tokenizer.texts_to_sequences(X_test)
X_test_sequences= sequence.pad_sequences(sequences,maxlen=max_len)

print("loss and accruacy",model.evaluate(X_test_sequences,Y_test))


y_pred=list(np.argmax(model.predict(X_test_sequences),1))
y_true=list(np.argmax((Y_test),1))
print('classification report for test set')
print(classification_report(y_true,y_pred))




##test for sa single sample
## add any devnagri sentence
text_data=[' इसे कार्ट में मत डालो  ']
sequences = tokenizer.texts_to_sequences(text_data)
X_text_data_sequences= sequence.pad_sequences(sequences,maxlen=max_len)
model.predict(X_text_data_sequences)
inp = model.input                                           
outputs = [layer.output for layer in model.layers][1:]         
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
test = X_text_data_sequences[0][np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print ('attention_weights pre-paddded sequence\n','sentence:',text_data,"\nweights",list(layer_outs[2][0][1]))






