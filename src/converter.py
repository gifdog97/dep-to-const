import os
import pyconll
from pyconll.util import find_nonprojective_deps

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from const import source_path_mapping

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--ud_dir', default='../../../resource/ud-treebanks-v2.7')
parser.add_argument('--language', default='English')
parser.add_argument('--convert_method', default='flat')
parser.add_argument('--output_source_dir', default='../../../resource/ud-converted')

def find_multiword(sentence):
    for token in sentence:
        if token.is_multiword():
            return False
    return True

def get_token_with_id(sentence, token_id):
    for token in sentence:
        if str(token_id) == token.id:
            return token

def extract_head(sentence):
    for token in sentence:
        if token.deprel == 'root':
            return token

def extract_children(sentence, parent_token) -> list:
    child_list = [int(parent_token.id)]
    for token in sentence:
        if token.head == parent_token.id:
            child_list.append(int(token.id))
    return sorted(child_list)

# convert all the parens into -LRB- or -RRB- to resolve ambiguity of phrase structure.
def create_leaf(token):
    form = token.form
    form = form.replace('(', '-LRB-')
    form = form.replace(')', '-RRB-')
    if token.upos == 'PUNCT':
        return f'({form} {form}) '
    return f'({token.upos} {form}) '

def extract_constituency(sentence, token) -> list:
    children = extract_children(sentence, token)
    if len(children) == 1:
        return create_leaf(token)
    constituency = f'({token.upos}P '
    for child_id in children:
        if child_id == int(token.id):
            form = token.form
            form = form.replace('(', '-LRB-')
            form = form.replace(')', '-RRB-')
            sub_constituency = f'({token.upos} {form}) '
        else:
            sub_constituency = extract_constituency(sentence, get_token_with_id(sentence, child_id))
        constituency += sub_constituency
    return constituency.rstrip() + ') '

def flat_converter(sentence: str) -> str:
    assert find_nonprojective_deps(sentence) == []
    head_token = extract_head(sentence)
    return extract_constituency(sentence, head_token).rstrip()

def main(args):
    file_type_list = ['train', 'dev', 'test']

    for file_type in file_type_list:
        source_path = source_path_mapping[args.language]
        path_to_corpus = os.path.join(args.ud_dir, source_path.format(file_type))
        if args.convert_method == 'flat':
            converter = flat_converter
        """
        elif args.convert_method == 'left':
            converter = left_converter
        elif args.convert_method == 'right':
            converter = right_converter
        """
        output_dir = os.path.join(args.output_source_dir, source_path.split('/')[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f'Converting {args.language} {file_type} corpus using {args.convert_method} converter.')
        corpus = pyconll.load_from_file(path_to_corpus)

        oneword_count = 0
        nonproj_count = 0
        inclempty_count = 0

        with open(os.path.join(output_dir, f'{file_type}.txt'), 'w') as f:
            for sentence in corpus:
                if len(sentence) == 1:
                    oneword_count += 1
                    continue
                try:
                    phrase_structure = converter(sentence)
                    f.write(phrase_structure)
                    f.write('\n')
                except AssertionError:
                    nonproj_count += 1
                    continue
                except KeyError:
                    inclempty_count += 1
                    continue

        logger.info(f'Number of sentence: {len(corpus)}')
        logger.info(f'Number of oneword sentence: {oneword_count}')
        logger.info(f'Number of non-projective sentence: {nonproj_count}')
        logger.info(f'Number of sentence with empty node: {inclempty_count}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)