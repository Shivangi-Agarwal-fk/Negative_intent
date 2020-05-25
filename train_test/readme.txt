
For training file:
    
python train.py 

All customizable parameters are in  32  to 48 in train file.


training_file format:
sentence1,label
sentence2,label
sentence3,label







For prediction file:

    
python predict.py

All customizable parameters are in  29  to 33 in predict file.

input_file format:
sentence1
sentence2
sentence3

output_file format(one line for one sentence output):
[attention_weights label confidence_score]
[attention_weights label confidence_score]
[attention_weights label confidence_score]


