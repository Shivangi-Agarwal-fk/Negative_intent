
For server file:
    
python server.py --ipaddess --port_num --model_path --tokenizer_path

model_path='model_files/model.h5'
tokenizer_path='model_files/tokenizer.pickle'






For client file:
    
python server.py --ipaddess --port_num --input_file --output_file


sample input_file= 'input.txt'
It should contain 1 sentence in each line

sample output_file='output.txt'
