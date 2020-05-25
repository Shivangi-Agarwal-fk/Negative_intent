#!/usr/bin/env python
# coding: utf-8


from __future__ import division
from __future__ import print_function
import numpy as np
import zmq, sys, time, socket, re
from zmq.error import ZMQError
from collections import defaultdict
import json
import traceback
from keras.models import load_model
from utils import *
from keras.preprocessing import sequence
import pickle

import os
import sys
from keras import backend as K
import random



class Server(object):
    def __init__(self, ip,port,model_path,tokenizer_path):
        self.port =port
        self.ip=ip
        self.model, self.tokenizer=self.load_model(model_path,tokenizer_path)
        self.attention_function=self.attention_evaluater(self.model)
        
        self._create_zmq_server(ip,port)
        
    def _create_zmq_server(self, ip,port):
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind('tcp://' + ip + ':' + str(port))
            print(' done socket binding')
          
        
    def load_model(self,model_path,tokenizer_path):
            model=load_model(model_path, custom_objects={'Attention': Attention})
            with open(tokenizer_path, 'rb') as handle:
                tok = pickle.load(handle)
            print('model_loaded')
            return model,tok
        
    def attention_evaluater(self,model):
            inp = model.input                                           # input placeholder
            outputs = [layer.output for layer in model.layers][1:]          # all layer outputs
            functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
            return functors
            
    def _extract_word_lists(self, input_word_lists):

        out_sentences = []
        for word_list_dict in input_word_lists:
            word_list = word_list_dict["word_list"]
            out_sentences.append(word_list)
        return out_sentences
    
    def _create_output_json(self, status, err_string):
        out_json = None
       
        out_json = {
            "version": 1,
            "header":{
                "status": status,
                "err_msg": err_string
            },
            "payload":{
                "sentences":[
                ]
            }
        }
        return out_json
    
    
    def find_output(self,input_sentences):
        
        data=clean_data(input_sentences)
        x=sentence_to_embedding(data,self.tokenizer)
        label,conf_score,attention_score=model_predict(x,self.model,self.attention_function)
        output= process_output( input_sentences,label,conf_score,attention_score)    
        return output
    
    def start(self):
        
        
        while True:
            
            try:
                api_call = self.socket.recv()
                start = time.time()
                params = json.loads(api_call)
    #             print(params,'params')
                input_sentences =  self._extract_word_lists(params['payload']['sentences'])
    #             print(input_sentences,'input')
                enc_input_sentences=input_sentences
                output= self.find_output(enc_input_sentences)
                out_json = self._create_output_json( 1, "ok")

                for idx, sentence in enumerate(output):
                    w_out = {"word_list": output[idx]['word_list'], "negative_intent": output[idx]['negative_intent'], "score": output[idx]['score']}
                    out_json["payload"]["sentences"].append(w_out)

                out_json_str = json.dumps(out_json,ensure_ascii=False) 
                print(out_json_str,'output')
                self.socket.send_string(out_json_str)
                sent_time = time.time()
                
            except:
                
                continue

           

if __name__ == '__main__':
    
        arguments =  sys.argv
        ipaddr=arguments[1]
        port=arguments[2]
        model_path=arguments[3]
        tokenizer_path=arguments[4]
        
        exp = Server(ipaddr,port,model_path,tokenizer_path)
        
        
        # port=8881
        # ipaddr='127.0.0.1'
        # exp = Server(ipaddr,port,'model_files/model.h5',"model_files/tokenizer.pickle")
        
        exp.start()
        
        
        
    




