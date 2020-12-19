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
            sub_constituency = f'({token.upos} {token.form}) '
        else:
            sub_constituency = extract_constituency(sentence, get_token_with_id(sentence, child_id))
        constituency += sub_constituency
    return constituency.rstrip() + ') '

def flat_converter(sentence: str) -> str:
    assert find_nonprojective_deps(sentence) == []
    head_token = extract_head(sentence)
    return extract_constituency(sentence, head_token).rstrip()

if __name__ == '__main__':
    language = 'English'
    path_to_ud = '../../../resource/ud-treebanks-v2.7'
    path_to_corpus = os.path.join(path_to_ud, source_path_mapping[language])

    logger.info(language)

    corpus = pyconll.load_from_file(path_to_corpus)

    nonproj_count = 0
    inclempty_count = 0

    for sentence in corpus:
        try:
            flat_structure = flat_converter(sentence)
            print(flat_structure)
        except AssertionError:
            nonproj_count += 1
            continue
        except KeyError:
            inclempty_count += 1
            continue

    logger.info(f'Number of sentence: {len(corpus)}')
    logger.info(f'Number of non-projective sentence: {nonproj_count}')
    logger.info(f'Number of sentence with empty node: {inclempty_count}')

