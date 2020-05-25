#!/usr/bin/env python
# coding: utf-8




import numpy as np
import zmq, sys, time, socket, re
from zmq.error import ZMQError
from collections import defaultdict
import json
import traceback
import argparse



class entity_zmq(object):
    def __init__(self, ipaddr, port, timeout=5000):
        self.ipaddr = ipaddr
        self.port = port
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.connect("tcp://%s:%s" % (self.ipaddr, self.port))
      
        
        
    def _process_output_json_negative_intent(self, out_json):
    
        sentences = out_json["payload"]["sentences"]
        entities = []
        for sentence in sentences:
            words_list = sentence["word_list"]
            attn = []
            for words in words_list:
                attn.append(words["attention_score"])
            attn.append(sentence["negative_intent"])
            attn.append(sentence["score"])
                
            entities.append(' '.join(attn))
            
            
        
        return entities

    

    def _create_default_input_json(self, mode):
        input_json = {
            "version": 1,
            "header":{
                "mode": mode
            },
            "payload":{
                "sentences":[

                ]
            }
        }
        return input_json

    def _create_json_from_sentences(self, sentences, mode):
        input_json = self._create_default_input_json(mode)
        for sentence in sentences:
            words_list = sentence.split()
            word_list_dict = {"word_list": words_list}
            input_json["payload"]["sentences"].append(word_list_dict)
        
        return input_json
    
    
    def send_to_negative_intent_classification_server(self, sentences_list, mode):
    
        input_json = self._create_json_from_sentences(sentences_list, mode)

        input_json_string = json.dumps(input_json, ensure_ascii=False)
        combined_line = ' '.join(sentences_list)

        zmq_send_time = time.time()
        try:

            self.socket.send_string(input_json_string)
         
        except:
            print ("couldnt send data", combined_line)

        opstring_list = []

        try:
            received_ouput=(self.socket.recv())
            opstring=json.loads(received_ouput)
            
            status = opstring["header"]["status"]
            if status == 1:
                output = ""
                if (mode=='negative_intent'):
                    
                    output = self._process_output_json_negative_intent(opstring)
               
                return output
            else:
                print ("Error status returned for:", combined_line)
                print ("Error message: {}".format(opstring["header"]["err_msg"]))
                return ["$INVALID$"]

        except zmq.error.Again:
            print ("No response received for:", combined_line)
            opstring_list = ["$INVALID$"]
            self.socket.close(0)
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.connect("tcp://%s:%s" % (self.ipaddr, self.port))
        except:
            traceback.print_exc()
            print ("Malformed response received for:", combined_line)
            opstring_list = ["$INVALID$"]
        return opstring_list




if __name__ == "__main__":
#     arguments =  sys.argv
#     ipaddr=arguments[1]
#     port=arguments[2]
#     filename=arguments[3]
    

    ##testing by single sentence
    ipaddr = '127.0.0.1'
    start = time.time()
    port = 8881
    obj = entity_zmq(ipaddr, port)
    test_sentence = ["पिंक वाला  पीला वाला आता चाहिए    "]
    outputsent = obj.send_to_negative_intent_classification_server(test_sentence, 'negative_intent')
    print('completed')
    print(test_sentence)
    print('output',outputsent)
    f = open("output.txt","w")
    print(outputsent, file=f)
    
#     filename="input.txt"

#     if filename != "":
#         infile = open(filename, 'r')
#         line_list = []
        
#         for line_read in infile:
#             line_read = line_read.strip()
#             line_list.append(line_read)
#         start = time.time()
#         outputsent = obj.send_to_negative_intent_classification_server(line_list, 'negative_intent')
#         end = time.time()
        
#         f = open("output.txt","w")
#         print(outputsent, file=f)
#             # if outputsent != "$INVALID$":
#             #     print line_read, "\n", outputsent
        
#         print ("Time required for all the queries : "+str((end-start)*1000)+" ms")
#         print(outputsent)
#         print(len(outputsent))







