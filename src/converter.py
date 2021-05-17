import argparse
import os
from pathlib import Path
import pyconll
from pyconll.util import find_nonprojective_deps
from nltk.tree import *
import unicodedata

from logging import getLogger, FileHandler, Formatter, DEBUG
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.propagate = False

parser = argparse.ArgumentParser()

# directory parameters
parser.add_argument('--source_path', default='path/to/source')
parser.add_argument('--output_path', default='path/to/output')

# method specification parameters
parser.add_argument('--convert_method', default='flat')
parser.add_argument('--without_label', action='store_true')
parser.add_argument('--use_pos_label', action='store_true')
parser.add_argument('--use_merged_pos_label', action='store_true')
parser.add_argument('--use_dep_label', action='store_true')

# other parameter(s)
parser.add_argument('--dev_test_sentence_num', default=5000)
parser.add_argument('--train_token_num', default=10000000)


class NonProjError(Exception):
    pass


class RootNonProjError(Exception):
    pass


class CFContainedError(Exception):
    pass


class ContainNoneError(Exception):
    pass


# Since pyconll package sometimes fail to censor non-projective dependency tree that contains crossing above root edge,
# function for handling this exception is defined here
def rootcross_included(sentence):
    root_id = int(extract_head(sentence).id)

    def is_crossing_root(token_id, head_id):
        return (token_id < root_id
                and root_id < head_id) or (token_id > root_id
                                           and root_id > head_id)

    for token in sentence:
        head_id = token.head
        if is_crossing_root(int(token.id), int(head_id)):
            return True

    return False


def Cf_included(s):
    if s is None:
        raise ContainNoneError
    for c in s:
        if unicodedata.category(c) == "Cf":
            return True
    return False


def sentence_to_str(sentence):
    s = ""
    for token in sentence:
        s += token.form
        s += " "
    return s.rstrip()


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


# Convert all the parens into -LRB- or -RRB- to resolve ambiguity of phrase structure.
# Remove sentence including control character because preprocess.py in batched-RNNG does not expect that.
# Remove space in form because preprocess.py expect each forms do not contain any space.
def sanitize_form(form):
    if Cf_included(form):
        raise CFContainedError
    if ' ' in form:
        logger.info(f'Space included in the form: {form}')
    return form.replace('(', '-LRB-').replace(')', '-RRB-').replace(
        '（', '-LRB-').replace('）', '-RRB-').replace(' ', '')


def create_leaf(form, nt):
    return f'({nt} {sanitize_form(form)}) '


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
            sub_constituency = f'({get_nt(token)} {sanitize_form(token.form)}) '
        else:
            sub_constituency = flat_converter(
                sentence, get_token_with_id(sentence, child_id), get_nt)
        constituency += sub_constituency
    return constituency.rstrip() + ') '


def make_phrase_from_left(sentence, token, left_children_ids,
                          right_children_ids, get_nt):
    if left_children_ids == []:
        if right_children_ids == []:
            return Tree(get_nt(token), [sanitize_form(token.form)])
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
            return Tree(get_nt(token), [sanitize_form(token.form)])
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
    if len(find_nonprojective_deps(sentence)) != 0:
        raise NonProjError
    if rootcross_included(sentence):
        raise RootNonProjError
    head_token = extract_head(sentence)
    return converter(sentence, head_token, get_nt).rstrip()


def generate_tokens(sentence):
    plain_sentence = ""
    for token in sentence:
        plain_sentence += sanitize_form(token.form) + ' '
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


def extract_data_type(conllu_file_name):
    if "train" in conllu_file_name:
        return "train"
    elif "dev" in conllu_file_name:
        return "dev"
    elif "test" in conllu_file_name:
        return "test"
    # this case (dataset is not train/dev/test) should be treated carefully
    return "other"


def find_conllu_files(source_path):
    file_list = []
    for conllu_file in Path(source_path).glob('**/*.conllu'):
        file_info = {}
        file_info['path'] = conllu_file
        file_info['data_type'] = extract_data_type(conllu_file.name)
        file_list.append(file_info)
    return file_list


def generate_path_info(args):
    files_to_convert = find_conllu_files(args.source_path)
    method_str = get_method_str(args)
    return files_to_convert, method_str, os.path.join(args.output_path,
                                                      method_str)


def main(args):
    conllu_files_to_convert, method_str, output_dir = generate_path_info(args)

    converter, get_nt = setup_functions(args)

    for conllu_file in conllu_files_to_convert:
        logger.info(
            f'Converting {conllu_file["path"].name} with {method_str} method.')
        corpus = pyconll.load_from_file(str(conllu_file["path"]))

        processed_sentence_num = 0
        token_num = 0

        inclempty_count = 0
        cfcontained_count = 0
        contain_none_count = 0
        nonproj_count = 0
        root_nonproj_count = 0

        with open(os.path.join(output_dir, f'{conllu_file["data_type"]}.txt'),
                  'w') as f:
            with open(
                    os.path.join(output_dir,
                                 f'{conllu_file["data_type"]}.tokens'),
                    'w') as g:
                for i, sentence in enumerate(corpus):
                    if i % 10000 == 0:
                        logger.info(f'{i} data has been converted.')
                    try:
                        phrase_structure = general_converter(
                            converter, sentence, get_nt)
                        if len(sentence) == 1:
                            phrase_structure = f'({get_nt(sentence[0])} {phrase_structure})'
                        f.write(phrase_structure)
                        f.write('\n')
                        g.write(generate_tokens(sentence))
                        g.write('\n')
                        processed_sentence_num += 1
                        token_num += len(sentence)
                        # extract 5000 sentence for dev/test set
                        if conllu_file[
                                "data_type"] != "train" and processed_sentence_num == args.dev_test_sentence_num:
                            break
                        if conllu_file[
                                "data_type"] == "train" and token_num == args.train_token_num:
                            break
                    except KeyError:
                        inclempty_count += 1
                        continue
                    except ContainNoneError:
                        contain_none_count += 1
                        continue
                    except NonProjError:
                        nonproj_count += 1
                        continue
                    except RootNonProjError:
                        root_nonproj_count += 1
                        continue
                    except CFContainedError:
                        logger.info(
                            f'Cf contained in {sentence_to_str(sentence)}')
                        cfcontained_count += 1
                        continue

        logger.info(f'Number of sentence: {len(corpus)}')
        logger.info(f'Number of tokens: {token_num}')
        logger.info(f'converted: {processed_sentence_num}')

        logger.info(f'Number of non-projective sentence: {nonproj_count}')
        logger.info(
            f'Number of root-non-projective sentence: {root_nonproj_count}')
        logger.info(f'Number of sentence with None: {contain_none_count}')
        logger.info(f'Number of sentence with empty node: {inclempty_count}')
        logger.info(
            f'Number of sentence with control character: {cfcontained_count}')


if __name__ == '__main__':
    args = parser.parse_args()
    source_path, method_str, output_dir = generate_path_info(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    handler = FileHandler(filename=f'{output_dir}/convert.log')
    handler.setLevel(DEBUG)
    handler.setFormatter(Formatter(fmt))
    logger.addHandler(handler)

    main(args)
