import os
import json
import argparse
import codecs
import sys
import time

from timebudget import timebudget
from multiprocessing import Process, Pool, freeze_support
import torch
from transformers import pipeline

"""
This code runs only on CPU.
"""

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

def translator(content, translate_pipe):
    if len(content) > 256:
        content = content.split('. ')
        parts = []
        for part in content:
            parts.append(translate_pipe(part)[0]['translation_text'])
        return '. '.join(parts)
    return translate_pipe(content)[0]['translation_text']

def run_translate(start, count):
    translated = []
    for index in range(start, start + count):
        row = {}
        for key in param_keys:
            if params[index][key] == '':
                row[key] = ''
                continue
            row[key] = translator(params[index][key], translate_pipe)
        translated.append(row)
    
    file_name = './output/output_{}_{}.json'.format(start, count)
    with open(file_name, "w", encoding="utf-8") as temp:
        temp.write(json.dumps(translated, ensure_ascii=False))

def get_pool_args(start, count):
    return {
        'params': params, 
        'param_keys': param_keys, 
        'translator': translator,
        'translate_pipe': translate_pipe,
        'start': start,
        'count': count
    }

if __name__ == '__main__':

    settings = {
        'params': params, 
        'param_keys': param_keys, 
        'translator': translator,
        'translate_pipe': translate_pipe,
        'start': 0,
        'count': 1
    }

    start_time = time.perf_counter()
    with Pool(4) as pool:
      result = pool.starmap(run_translate, [
          (0, 5),
          (5, 10),
          (10, 15),
          (15, 20)
      ])
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
    # p = Process(target=run_translate, args=([{
    #     'params': params, 
    #     'param_keys': param_keys, 
    #     'translator': translator,
    #     'translate_pipe': translate_pipe,
    #     'start': 0,
    #     'count': 1
    # }]))
    # p.start()
    # p.join()
    
    # p2 = Process(target=run_translate, args=([{
    #     'params': params, 
    #     'param_keys': param_keys, 
    #     'translator': translator,
    #     'translate_pipe': translate_pipe,
    #     'start': 1,
    #     'count': 2
    # }]))
    # p2.start()
    # p2.join()
