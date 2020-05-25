import numpy as np 
import re
from keras import initializers
from keras import backend as K
from keras.engine import InputSpec, Layer
import random
from keras.preprocessing import sequence
import pickle
import string



def dot_product(x, kernel):
       return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
       
class Attention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('normal')
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='W')
        
        self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='b')

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='u')
#         self.trainable_weights = [self.W,self.u]
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        product = dot_product(x, self.W)
        product += self.b
        product = K.tanh(product)
        product_new = dot_product(product, self.u)
        att = K.exp(product_new)
        if mask is not None:            
            att *= K.cast(mask, K.floatx())
        att /= K.cast(K.sum(att, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        att = K.expand_dims(att)
        weighted_input = x * att
        return [K.sum(weighted_input, axis=1),att]

    def compute_output_shape(self, input_shape):

        return [(input_shape[0], input_shape[-1]),(input_shape[0], input_shape[1])]



def remove_punc(sentence):
    
    punct_set= set(string.punctuation)#Saving punctuation into a set
    s = ''.join(ch for ch in sentence if ch not in punct_set)
    return s


def clean_data(sentences): 
    clean_sentences=[]
   
    for sent in sentences:
      
        senten=''
        for word in sent:
            senten=senten+' '+word
        senten=remove_punc(senten)
        clean_sentences.append(senten)

    return clean_sentences


def sentence_to_embedding(sentences,tokenizer):
    # print(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=15)
    
    return sequences_matrix



def model_predict(x,model,functors):
    
    res=model.predict(x)
    conf_score=[max(i) for i in res]
    out_label=list(np.argmax(res,1))
    label=['positive' if i== 0 else 'negative' for i in out_label ]
    
    attention_score=[]
    
    for i in range(len(x)):
        
        test = x[i][np.newaxis,...]
        layer_outs = [func([test, 1.]) for func in functors]
        attntn=layer_outs[2][0][1][0]
        att=[]
        for  i in attntn:
            att.append(i[0])
        attention_score.append(att)
    
    
    
    return label,conf_score,attention_score


def process_output( sentences,label,conf_score,attention_score):
    
    output=[]
    for index,sent in enumerate(sentences):
        result={} 
        word_lis=[]
        sent.reverse()
        attention_score[index].reverse()
        for ind,word in enumerate(sent):
            word_dict={}
            word_dict['word']=word
            word_dict['attention_score']=str("%.2f" %(attention_score[index][ind]))
            word_lis.append(word_dict)
          
        word_lis.reverse()
        result['negative_intent']=label[index]
        result['score']=str("%.2f" %conf_score[index])
        result['word_list']=word_lis
        output.append(result)

    return output




    