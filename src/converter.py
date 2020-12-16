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

def flat_converter(sentence: str) -> str:
    assert find_nonprojective_deps(sentence) == []
    return ""

if __name__ == '__main__':
    language = 'Chinese'
    path_to_ud = '../../../resource/ud-treebanks-v2.7'
    path_to_corpus = os.path.join(path_to_ud, source_path_mapping[language])

    logger.info(language)

    corpus = pyconll.load_from_file(path_to_corpus)

    nonproj_count = 0
    inclempty_count = 0

    for sentence in corpus:
        try:
            flat_structure = flat_converter(sentence)
        except AssertionError:
            nonproj_count += 1
            continue
        except KeyError:
            inclempty_count += 1
            continue

    logger.info(f'Number of sentence: {len(corpus)}')
    logger.info(f'Number of non-projective sentence: {nonproj_count}')
    logger.info(f'Number of sentence with empty node: {inclempty_count}')

