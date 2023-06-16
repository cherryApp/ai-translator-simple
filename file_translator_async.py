import os
import json
import argparse
import codecs
import sys
import time
import math
import asyncio
import tqdm
import nltk.data
from functools import reduce

from timebudget import timebudget
from multiprocessing import Process, Pool, freeze_support
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
parser.add_argument(
    '-c',
    '--cudacore',
    type=int, 
    help="index of cuda core",
    nargs='?', 
    default=0
)
parser.add_argument(
    '-s',
    '--start',
    type=int, 
    help="start position in the dataset",
    nargs='?', 
    default=0
)
parser.add_argument(
    '-n',
    '--num',
    type=int, 
    help="number of data from the dataset",
    nargs='?', 
    default=100
)
parser.add_argument(
    '-cs',
    '--chunk_size',
    type=int, 
    help="size of chunks of text",
    nargs='?', 
    default=512
)
parser.add_argument(
    '-b',
    '--batch_size',
    type=int, 
    help="model batching size",
    nargs='?', 
    default=8
)

args = parser.parse_args()

# Optimize to mac m1
pipes = []
if args.device == 'cuda':
    for index in range( torch.cuda.device_count() ):
        pipes.append(
            pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-tc-big-en-hu",
                device=torch.device('cuda:'+str(index))
            )
        )
else:
    device = torch.device(args.device)
    pipes.append(
        pipeline(
            "translation", 
            model="Helsinki-NLP/opus-mt-tc-big-en-hu", 
            device=device
        )
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

async def run_translate_batch(start, count):
    translated = []
    pipe = 0
    for index in range(start, start + count):
        row = {}
        for key in param_keys:
            if params[index][key] == '':
                row[key] = ''
                continue
            # if len(pipes) > 1:
                # pipe ^= 1
            row[key] = translator(params[index][key], pipes[pipe])
        translated.append(row)
    
    file_name = './output/output_{}_{}.json'.format(start, count)
    with open(file_name, "w", encoding="utf-8") as temp:
        temp.write(json.dumps(translated, ensure_ascii=False))

def splitter(size = 1000, datarow = {}):
    splitted = []
    base = {'instruction': '', 'input': '', 'output': ''}
    if len(datarow['instruction']) > size:
        splitted.append(base.copy())
        splitted.append(base.copy())
        sentences = datarow['instruction'].split('. ')
        splitted[0]['instruction'] = '. '.join(sentences[0:math.floor(len(sentences)/2)])
        splitted[1]['instruction'] = '. '.join(sentences[math.floor(len(sentences)/2):])


def preprocessor(raw_text = '', size=1000):
    if len(raw_text) <= size:
        return [{'input': raw_text}]
    
    parts = []
    buffer = []
    output_length = 0
    sentences = sentence_tokenizer.tokenize(raw_text.strip())
    for i in range(0, len(sentences)):
        if output_length > size:
            parts.append({'input': ''.join(buffer)})
            output_length = 0
            buffer = []
        buffer.append(sentences[i])
        output_length += len(sentences[i])
    
    if len(buffer) > 0:
        parts.append({'input': ' '.join(buffer)} )
    
    return parts


def run_translate(start, count, pipe=0, batch_size=8, chunk_size=512):
    dataset = params[start:(start + count)]

    translated = []

    shapes = []

    output_list = []

    flat_dataset = []

    file_name = './output/output_{}_{}.json'.format(start, count)

    for row in dataset:
        for key in param_keys:
            processed = preprocessor(row[key], chunk_size)
            shapes += [[key, len(processed)]  for x in range(0, len(processed))]
            flat_dataset += processed

    with tqdm(total=len(flat_dataset), desc='Rows({}): '.format(len(flat_dataset))) as pbar:
        for out in pipes[pipe](KeyDataset(flat_dataset, "input"), batch_size=batch_size): 
            translated.append(out)
            pbar.update(1)


    i = 0
    row = {'instruction': '', 'input': '', 'output': ''}
    while i < len(translated):
        for j in range(i, i + shapes[i][1]):
            row[shapes[i][0]] += translated[j][0]['translation_text'] + ' '
        i += shapes[i][1]
        
        if shapes[j][0] == 'output':
            output_list.append(row)
            row = {'instruction': '', 'input': '', 'output': ''}

    with open(file_name, "w", encoding="utf-8") as temp:
        temp.write(json.dumps(output_list, ensure_ascii=False))

    return True

def run_translate_old(start, count, pipe = 0, batch_size = 8, chunk_size=512):

    dataset = params[start:(start + count)]

    translated = []

    instructions = []

    inputs = []

    outputs = []

    file_name = './output/output_{}_{}.json'.format(start, count)

    for out in tqdm(
        pipes[pipe](KeyDataset(dataset, "instruction"), batch_size=batch_size, max_length=1024),
        desc='Instructions({}): '.format(len(dataset))
    ):
        instructions.append(out)

    for out in tqdm(
        pipes[pipe](KeyDataset(dataset, "input"), batch_size=batch_size, max_length=1024),
        desc='Inputs({}): '.format(len(dataset))
    ):
        inputs.append(out)

    for out in tqdm(
        pipes[pipe](KeyDataset(dataset, "output"), batch_size=batch_size, max_length=1024),
        desc='Outputs({}): '.format(len(dataset))
    ):
        outputs.append(out)

    for index in range(0, count):
        row = {}
        try:
            row['instruction'] = instructions[index][0]["translation_text"]
        except IndexError:
            row['instruction'] = ''

        try:
            row['input'] = inputs[index][0]["translation_text"]
        except IndexError:
            row['input'] = ''

        try:
            row['output'] = outputs[index][0]["translation_text"]
        except IndexError:
            row['output'] = ''

        translated.append(row)

    with open(file_name, "w", encoding="utf-8") as temp:
        temp.write(json.dumps(translated, ensure_ascii=False))

    return True

if __name__ == '__main__':
    # print(params[16382:16384])
    # sys.exit()

    print("Start translate: ")
    print("Device: ", args.device)
    print("Cuda device: ", args.cudacore)
    print("Start: ", args.start)
    print("Num of data: ", args.num)
    print("Batch size: ", args.batch_size)
    print("Chunk size: ", args.chunk_size, "\n")

    start_time = time.perf_counter()

    core = 0
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        core = args.cudacore

    run_translate(args.start, args.num, core, args.batch_size, args.chunk_size)

    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")

"""
Here is an implementation of a red-black tree in C++:\n\n```\n#include <iostream>\n\nusing namespace std;\n\nenum Color {RED, BLACK};\n\nstruct Node {\n    int val;\n    Color color;\n    Node *left, *right, *parent;\n    \n    Node() {}\n    \n    Node(int v, Color c) {\n        val = v;\n        color = c;\n        left = right = parent = NULL;\n    }\n};\n\nclass RedBlackTree {\n    private:\n        Node *root;\n        \n    public:\n        RedBlackTree() { root = NULL; }\n        \n        /* Helper functions */\n        Node* grandparent(Node *n) {\n            if (n && n->parent) { return n->parent->parent; }\n            else { return NULL; }\n        }\n        \n        Node* uncle(Node *n) {\n            Node *g = grandparent(n);\n            if (!g) { return NULL; }\n            if (n->parent == g->left) { return g->right; }\n            else { return g->left; }\n        }\n        \n        Node* sibling(Node *n) {\n            if (!n || !n->parent) { return NULL; }\n            if (n == n->parent->left) { return n->parent->right; }\n            else { return n->parent->left; }\n        }\n        \n        void fix_tree(Node *n) {\n            while (n != root && n->color == RED && n->parent->color == RED) {\n                Node *p = n->parent;\n                Node *g = grandparent(n);\n                Node *u = uncle(n);\n                \n                if (u && u->color == RED) {\n                    p->color = BLACK;\n                    u->color = BLACK;\n                    g->color = RED;\n                    n = g;\n                } else {\n                    if (p == g->left) {\n                        if (n == p->right) {\n                            n = p;\n                            left_rotate(n);\n                        }\n                        p->color = BLACK;\n                        g->color = RED;\n                        right_rotate(g);\n                    } else {\n                        if (n == p->left) {\n                            n = p;\n                            right_rotate(n);\n                        }\n                        p->color = BLACK;\n                        g->color = RED;\n                        left_rotate(g);\n                    }\n                }\n            }\n            root->color = BLACK;\n        }\n        \n        void left_rotate(Node *n) {\n            Node *r = n->right;\n            n->right = r->left;\n            if (r->left) { r->left->parent = n; }\n            r->parent = n->parent;\n            if (!n->parent) { root = r; }\n            else if (n == n->parent->left) { n->parent->left = r; }\n            else { n->parent->right = r; }\n            r->left = n;\n            n->parent = r;\n        }\n        \n        void right_rotate(Node *n) {\n            Node *l = n->left;\n            n->left = l->right;\n            if (l->right) { l->right->parent = n; }\n            l->parent = n->parent;\n            if (!n->parent) { root = l; }\n            else if (n == n->parent->left) { n->parent->left = l; }\n            else { n->parent->right = l; }\n            l->right = n;\n            n->parent = l;\n        }\n        \n        void insert_node(int val) {\n            Node *n = new Node(val, RED);\n            if (!root) { root = n; }\n            else {\n                Node *current = root;\n                Node *parent = NULL;\n                \n                while (current) {\n                    parent = current;\n                    if (val < current->val) { current = current->left; }\n                    else { current = current->right; }\n                }\n                n->parent = parent;\n                if (val < parent->val) { parent->left = n; }\n                else { parent->right = n; }\n                fix_tree(n);\n            }\n        }\n        \n        Node* search_node(int val) {\n            Node *current = root;\n            while (current && current->val != val) {\n                if (val < current->val) { current = current->left; }\n                else { current = current->right; }\n            }\n            return current;\n        }\n        \n        void inorder(Node *n) {\n            if (!n) { return; }\n            inorder(n->left);\n            cout << n->val << \" \";\n            inorder(n->right);\n        }\n        \n        void print_tree() {\n            inorder(root);\n            cout << endl;\n        }\n};\n```\n\nThis implementation includes the insertion of nodes, searching for nodes, and printing the tree using an inorder traversal. The fix_tree function is responsible for maintaining the red-black balance after each insert.
"""