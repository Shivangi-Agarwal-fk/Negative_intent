#!/usr/bin/env python
# coding: utf-8


import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, concatenate, Dropout, Activation, Input
from keras.callbacks import EarlyStopping
import re
from sklearn.preprocessing import LabelEncoder
from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras import constraints
from keras import optimizers
from keras import regularizers
from keras.engine import InputSpec, Layer
from attention import *
import os
import pickle
from keras import backend as K
import string
from keras.models import load_model



input_file='input_for_prediction.txt'                          # input query file
model_path='model_files/model.h5'               # model path to load model
tokenizer_path='model_files/tokenizer.pickle'   # tokenizer path to load tokenizer
max_len = 15                                    # max sequence len
output_file='prediction_output.txt'                    # output file


def remove_punc(sentence):
    
    punct_set= set(string.punctuation)
    s = ''.join(ch for ch in sentence if ch not in punct_set)
    return s

def preprocess(filename):
    data = open(filename)
    with data as f:
        lines = [line.rstrip() for line in f]
        query=[li.strip() for li in lines]
        query=[remove_punc(li) for li in query]

    return query


## preprocess data
data=preprocess(''+input_file)

#load tokenizer
with open(tokenizer_path, 'rb') as handle:
    tok = pickle.load(handle)

## convert to sequence
sequences = tok.texts_to_sequences(data)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

##load model
model=load_model(model_path, custom_objects={'Attention': Attention})


res=model.predict(sequences_matrix)
conf_score=[max(i) for i in res]
out_label=list(np.argmax(res,1))
label=['positive' if i== 0 else 'negative' for i in out_label ]


##for finding attention  from internal layers
attention_score=[]
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    
    
for i in range(len(sequences_matrix)):

    test = sequences_matrix[i][np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    attntn=layer_outs[2][0][1][0]
    att=[]
    for  i in attntn:
        att.append(i[0])
    attention_score.append(att)
    
       
## output in form [attention_weights label confscore]
data=[x.split() for x in data]
result=[]
for index,sent in enumerate(data):
    
    word_lis=[]
    sent.reverse()
    attention_score[index].reverse()
    for ind,word in enumerate(sent):
      
     
        word_lis.append(str("%.3f" %(attention_score[index][ind])))
    
    word_lis.append(label[index])
    word_lis.append(str("%.3f" %conf_score[index]))
    
    result.append(word_lis)
        

print(result)
## save output
with open('prediction.txt', 'w') as f:
    for item in result:
        f.write("%s\n" % item)






