import os
import json
import argparse

from timebudget import timebudget
import torch
from transformers import pipeline

# Parse arguments and set default values
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    type=str, 
    help="enter json file path",
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
    default='./output.json'
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

# Optimize to mac m1
device = torch.device(args.device)
translate_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-hu", device=device)

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = args.file
abs_file_path = os.path.join(script_dir, rel_path)

json_file = open(abs_file_path)

content = json_file.read()

params = json.loads(content)

param_keys = ['instruction', 'input', 'output']

translated = []

def translator(content):
    if len(content) > 256:
        content = content.split('. ')
        parts = []
        for part in content:
            parts.append(translate_pipe(part)[0]['translation_text'])
        return '. '.join(parts)
    return translate_pipe(content)[0]['translation_text']

def run_translate(params, param_keys, translator):
    translated = []
    for index in range(10):
        row = {}
        for key in param_keys:
            if params[index][key] == '':
                row[key] = ''
                continue
            row[key] = translator(params[index][key])
        translated.append(row)

if __name__ == '__main__':
    run_translate(params, param_keys, translator)
