import numpy as np 
import re
from keras import initializers
from keras import backend as K
from keras.engine import InputSpec, Layer
import random


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
