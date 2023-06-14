import os
import json
import argparse
import codecs
import sys
import time
import asyncio

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
translate_pipe = pipeline(
    "translation", 
    model="Helsinki-NLP/opus-mt-tc-big-en-hu", 
    num_workers=16,
    batch_size=16,
    device=device
)

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = args.file
abs_file_path = os.path.join(script_dir, rel_path)

json_file = open(abs_file_path)

content = json_file.read()

params = json.loads(content)

param_keys = ['instruction', 'input', 'output']

translated = []

def translator(content, translate_pipe):
    current_len = 0
    if len(content) > 256:
        content = content.split('.')
        parts = []
        buffer = ''
        for part in content:
            if len(buffer) + len(part) > 256:
                parts.append(translate_pipe(buffer)[0]['translation_text'])
                buffer = ''
            else:
                buffer += part + '.'
        if len(buffer) > 0:
            parts.append(translate_pipe(buffer)[0]['translation_text'])
        return ''.join(parts)
    return translate_pipe(content)[0]['translation_text']

async def run_translate(start, count):
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
    start_time = time.perf_counter()
    
    loop = asyncio.get_event_loop()
    # for index in range(0, 180, 20):
    #     loop.create_task(run_translate(index, 20))
    loop.run_until_complete(run_translate(180, 20))

    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
