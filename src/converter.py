import argparse
import os
import pyconll
from pyconll.util import find_nonprojective_deps
from nltk.tree import *

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

source_path_mapping = {
    'English_EWT': 'UD_English-EWT/en_ewt-ud-{}.conllu',
    'Japanese_GSD': 'UD_Japanese-GSD/ja_gsd-ud-{}.conllu',
    'French_GSD': 'UD_French-GSD/fr_gsd-ud-{}.conllu',
    'German_GSD': 'UD_German-GSD/de_gsd-ud-{}.conllu',
    'Hebrew_HTB': 'UD_Hebrew-HTB/he_htb-ud-{}.conllu',
    'Russian_GSD': 'UD_Russian-GSD/ru_gsd-ud-{}.conllu',
    'ptb-to-ud': 'ptb-to-ud/{}.conllu'
}

parser = argparse.ArgumentParser()

# directory parameters
parser.add_argument('--ud_dir', default='../../../resource/ud-treebanks-v2.7')
parser.add_argument('--output_source_dir',
                    default='../../../resource/ud-converted')

# data to perform conversion
parser.add_argument('--language', default='English_EWT')

# method specification parameters
parser.add_argument('--convert_method', default='flat')
parser.add_argument('--without_label', action='store_true')
parser.add_argument('--use_pos_label', action='store_true')
parser.add_argument('--use_merged_pos_label', action='store_true')
parser.add_argument('--use_dep_label', action='store_true')


def get_token_with_id(sentence, token_id):
    for token in sentence:
        if str(token_id) == token.id:
            return token


def extract_head(sentence):
    for token in sentence:
        if token.deprel == 'root':
            return token


def extract_children(sentence, parent_token):
    child_list = [int(parent_token.id)]
    for token in sentence:
        if token.head == parent_token.id:
            child_list.append(int(token.id))
    return sorted(child_list)


def extract_left_children(sentence, parent_token):
    child_list = extract_children(sentence, parent_token)
    return list(filter(lambda c_id: c_id < int(parent_token.id), child_list))


def extract_right_children(sentence, parent_token):
    child_list = extract_children(sentence, parent_token)
    return list(filter(lambda c_id: c_id > int(parent_token.id), child_list))


# convert all the parens into -LRB- or -RRB- to resolve ambiguity of phrase structure.
def convert_parens(form):
    return form.replace('(', '-LRB-').replace(')', '-RRB-')


def create_leaf(form, nt):
    form = convert_parens(form)
    return f'({nt} {form}) '


def get_X_nt(token):
    return 'X'


def get_pos_nt(token):
    return f'{token.upos}P'


def get_merge_pos_nt(token):
    return get_pos_nt(token).replace('PRONP', 'NOUNP').replace(
        'PROPNP', 'NOUNP').replace('DETP', 'NOUNP')


def get_dep_nt(token):
    return f'{token.deprel}'


def flat_converter(sentence, token, get_nt):
    children = extract_children(sentence, token)
    if len(children) == 1:
        return create_leaf(token.form, get_nt(token))
    constituency = f'({get_nt(token)} '
    for child_id in children:
        if child_id == int(token.id):
            form = convert_parens(token.form)
            sub_constituency = f'({get_nt(token)} {form}) '
        else:
            sub_constituency = flat_converter(
                sentence, get_token_with_id(sentence, child_id), get_nt)
        constituency += sub_constituency
    return constituency.rstrip() + ') '


def make_phrase_from_left(sentence, token, left_children_ids,
                          right_children_ids, get_nt):
    if left_children_ids == []:
        if right_children_ids == []:
            return Tree(get_nt(token), [convert_parens(token.form)])
        else:
            r_token = get_token_with_id(sentence, right_children_ids.pop(-1))
            return Tree(get_nt(token), [
                make_phrase_from_left(sentence, token, left_children_ids,
                                      right_children_ids, get_nt),
                make_phrase_from_left(
                    sentence, r_token, extract_left_children(
                        sentence, r_token),
                    extract_right_children(sentence, r_token), get_nt)
            ])

    l_token = get_token_with_id(sentence, left_children_ids.pop(0))
    return Tree(get_nt(token), [
        make_phrase_from_left(
            sentence, l_token, extract_left_children(sentence, l_token),
            extract_right_children(sentence, l_token), get_nt),
        make_phrase_from_left(sentence, token, left_children_ids,
                              right_children_ids, get_nt)
    ])


def make_phrase_from_right(sentence, token, left_children_ids,
                           right_children_ids, get_nt):
    if right_children_ids == []:
        if left_children_ids == []:
            return Tree(get_nt(token), [convert_parens(token.form)])
        else:
            l_token = get_token_with_id(sentence, left_children_ids.pop(0))
            return Tree(get_nt(token), [
                make_phrase_from_right(
                    sentence, l_token, extract_left_children(
                        sentence, l_token),
                    extract_right_children(sentence, l_token), get_nt),
                make_phrase_from_right(sentence, token, left_children_ids,
                                       right_children_ids, get_nt),
            ])
    r_token = get_token_with_id(sentence, right_children_ids.pop(-1))
    return Tree(get_nt(token), [
        make_phrase_from_right(sentence, token, left_children_ids,
                               right_children_ids, get_nt),
        make_phrase_from_right(
            sentence, r_token, extract_left_children(sentence, r_token),
            extract_right_children(sentence, r_token), get_nt)
    ])


def left_converter(sentence, head_token, get_nt):
    return make_phrase_from_left(sentence, head_token,
                                 extract_left_children(sentence, head_token),
                                 extract_right_children(sentence, head_token),
                                 get_nt).pformat(margin=1e100)


def right_converter(sentence, head_token, get_nt):
    return make_phrase_from_right(sentence, head_token,
                                  extract_left_children(sentence, head_token),
                                  extract_right_children(sentence, head_token),
                                  get_nt).pformat(margin=1e100)


def general_converter(converter, sentence, get_nt):
    assert find_nonprojective_deps(sentence) == []
    head_token = extract_head(sentence)
    return converter(sentence, head_token, get_nt).rstrip()


def generate_tokens(sentence):
    plain_sentence = ""
    for token in sentence:
        plain_sentence += token.form + ' '
    return plain_sentence.rstrip()


def setup_functions(args):
    if args.convert_method == 'flat':
        converter = flat_converter
    elif args.convert_method == 'left':
        converter = left_converter
    elif args.convert_method == 'right':
        converter = right_converter

    if args.without_label:
        get_nt = get_X_nt
    elif args.use_pos_label:
        get_nt = get_pos_nt
    elif args.use_merged_pos_label:
        get_nt = get_merge_pos_nt
    elif args.use_dep_label:
        get_nt = get_dep_nt

    return converter, get_nt


def get_method_str(args):
    convert_method_str = args.convert_method
    if args.without_label:
        label_method_str = 'X'
    elif args.use_pos_label:
        label_method_str = 'POS'
    elif args.use_merged_pos_label:
        label_method_str = 'M_POS'
    elif args.use_dep_label:
        label_method_str = 'DEP'
    return f'{convert_method_str}-{label_method_str}'


def main(args):
    source_path = source_path_mapping[args.language]
    method_str = get_method_str(args)

    output_dir = os.path.join(args.output_source_dir,
                              source_path.split('/')[0], method_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_type_list = ['train', 'dev', 'test']

    converter, get_nt = setup_functions(args)

    for file_type in file_type_list:
        path_to_corpus = os.path.join(args.ud_dir,
                                      source_path.format(file_type))

        logger.info(
            f'Converting {args.language} {file_type} corpus with {method_str} method.'
        )
        corpus = pyconll.load_from_file(path_to_corpus)

        nonproj_count = 0
        inclempty_count = 0

        with open(os.path.join(output_dir, f'{file_type}.txt'), 'w') as f:
            with open(os.path.join(output_dir, f'{file_type}.tokens'),
                      'w') as g:
                for sentence in corpus:
                    try:
                        phrase_structure = general_converter(
                            converter, sentence, get_nt)
                        if len(sentence) == 1:
                            phrase_structure = f'({get_nt(sentence[0])} {phrase_structure})'
                        f.write(phrase_structure)
                        f.write('\n')
                        g.write(generate_tokens(sentence))
                        g.write('\n')
                    except AssertionError:
                        nonproj_count += 1
                        continue
                    except KeyError:
                        inclempty_count += 1
                        continue

        logger.info(f'Number of sentence: {len(corpus)}')
        logger.info(f'Number of non-projective sentence: {nonproj_count}')
        logger.info(f'Number of sentence with empty node: {inclempty_count}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
