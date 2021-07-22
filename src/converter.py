import os
from pathlib import Path
import pyconll
from pyconll.util import find_nonprojective_deps
from nltk.tree import *
import unicodedata

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
        if token.form is not None:
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


# 1. If a sentence includes control character, then raise Error because preprocess.py does not expect it. (This is not a sanitization, so ideally this procedure should be in other function.)
# 2. Convert all the parens into -LRB- or -RRB- to resolve ambiguity of phrase structure.
# 3. Remove space in form because preprocess.py expect each forms do not contain any space.
# - Space is contained at least in French_GSD, but they are numeric, therefore possibly not problematic.
def sanitize_form(form):
    if Cf_included(form):
        raise CFContainedError
    # if ' ' in form:
    #     logger.info(f'Space included in the form: {form}')
    return form.replace('(', '-LRB-').replace(')', '-RRB-').replace('（', '-LRB-').replace('）', '-RRB-').replace(' ', '')


def create_leaf(nt, form):
    return f'({nt} {sanitize_form(form)}) '


def create_leaf_with_Tree(nt, form):
    return Tree(nt, [sanitize_form(form)])


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
        return create_leaf(get_nt(token), token.form)
    constituency = f'({get_nt(token)} '
    for child_id in children:
        if child_id == int(token.id):
            sub_constituency = create_leaf(get_nt(token), token.form)
        else:
            sub_constituency = flat_converter(
                sentence, get_token_with_id(sentence, child_id), get_nt)
        constituency += sub_constituency
    return constituency.rstrip() + ') '


def make_phrase_from_left(sentence, token, left_children_ids,
                          right_children_ids, get_nt):
    if left_children_ids == []:
        if right_children_ids == []:
            return create_leaf_with_Tree(get_nt(token), token.form)
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
            return create_leaf_with_Tree(get_nt(token), token.form)
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


def find_conllu_files(source_path):
    return [conllu_file for conllu_file in Path(source_path).glob('**/*.conllu')]


def generate_path_info(args):
    files_to_convert = find_conllu_files(args.source_path)
    method_str = get_method_str(args)
    return files_to_convert, method_str, os.path.join(args.output_path,
                                                      method_str)
