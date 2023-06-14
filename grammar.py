from transformers import pipeline
import os, sys
import argparse
import nltk.data
import torch

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Parse arguments and set default values
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    type=str, 
    help="enter the file path to checking",
    nargs='?', 
    default='', 
    required=True
)
parser.add_argument(
    '-o',
    '--output',
    type=str, 
    help="enter output file path",
    nargs='?', 
    default='./grammer-done.txt'
)
parser.add_argument(
    '-d',
    '--device',
    type=str, 
    help="device for pytorch",
    nargs='?', 
    default='cpu'
)
args = parser.parse_args()

device = torch.device('cpu')
if torch.backends.mps.is_available():
    print('device: mps')
    device = torch.device('mps')
elif torch.backends.cuda.is_available():
    device = torch.device('cuda:0')
    print('device: cuda')

corrector = pipeline(
              'text2text-generation',
              'pszemraj/flan-t5-large-grammar-synthesis',
              device=device
              )

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = args.file
abs_file_path = os.path.join(script_dir, rel_path)

source_file = open(abs_file_path)

content = source_file.read()
source_file.close()

output = []
for sentence in sentence_tokenizer.tokenize(content.strip()):
    output.append(corrector(sentence)[0]['generated_text'])

if args.output is not None:
    output_file = open(
        os.path.join(script_dir, args.output),
        'w'
    )
    output_file.write('\n'.join(output))
    output_file.close()
else:
    print(output)
