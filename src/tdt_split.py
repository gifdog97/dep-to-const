"""
input:  conllu file
output: train / dev / test set
"""

import os
import argparse
import pyconll
from pathlib import Path

SENT = 80


def tdt_split(source_path, train_path, dev_path, test_path):
    # it's danger to load corpus as a whole...
    # corpus = pyconll.load_from_file(source_path)
    dev_test_sentences = SENT
    get_dev = True
    get_test = False
    sentence_num = 0
    with open(source_path, 'r') as inputconllu, \
         open(train_path, 'w')  as trainfile, \
         open(dev_path, 'w')    as devfile, \
         open(test_path, 'w')   as testfile:
        while True:
            conllu_str = ""
            while True:
                line = inputconllu.readline()
                conllu_str += line
                if line == "\n" or line == "":
                    break
            if line == "":
                break
            sentence = pyconll.load.load_from_string(conllu_str)[0]
            sentence_num += 1
            if get_dev:
                devfile.write(sentence.conll())
                devfile.write('\n\n')
                if sentence_num == dev_test_sentences:
                    sentence_num = 0
                    get_dev = False
                    get_test = True
            elif get_test:
                testfile.write(sentence.conll())
                testfile.write('\n\n')
                if sentence_num == dev_test_sentences:
                    sentence_num = 0
                    get_test = False
            else:
                trainfile.write(sentence.conll())
                trainfile.write('\n\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='en')
    parser.add_argument('--id', default='0')

    args = parser.parse_args()

    data_path = Path(f'/groups/gab50271/aac13111kt/resource/wiki-pre/split/LM_dataset/{args.lang}')
    source_path = Path(f'{data_path}/{args.id.zfill(2)}.conllu')
    train_path = Path(f'{data_path}/{args.id.zfill(2)}/train.conllu')
    dev_path = Path(f'{data_path}/{args.id.zfill(2)}/dev.conllu')
    test_path = Path(f'{data_path}/{args.id.zfill(2)}/test.conllu')

    train_path.parent.mkdir(parents=True, exist_ok=True)

    tdt_split(source_path, train_path, dev_path, test_path)

