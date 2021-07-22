import argparse
import os
from converter import *
import pyconll

from logging import getLogger, FileHandler, Formatter, DEBUG
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.propagate = False

parser = argparse.ArgumentParser()

# directory parameters
parser.add_argument('--source_path',
                    default='../../../resource/ud-treebanks-v2.7/English_EWT')
parser.add_argument('--output_path',
                    default='../../../resource/ud-converted')

# method specification parameters
parser.add_argument('--convert_method', default='flat')
parser.add_argument('--without_label', action='store_true')
parser.add_argument('--use_pos_label', action='store_true')
parser.add_argument('--use_merged_pos_label', action='store_true')
parser.add_argument('--use_dep_label', action='store_true')

# other parameter(s)
parser.add_argument('--dev_test_sentence_num', default=5000)
parser.add_argument('--train_token_num', default=40000000)
parser.add_argument('--write_deptree', action='store_true')


def convert_conllu_files(args):
    conllu_files_to_convert, method_str, output_dir = generate_path_info(args)
    converter, get_nt = setup_functions(args)
    original_deptree_dir = Path(os.path.join(output_dir, "original_deptree"))
    if not original_deptree_dir.exists():
        original_deptree_dir.mkdir()

    for conllu_file in conllu_files_to_convert:
        logger.info(f'Converting {conllu_file.name} with {method_str} method.')
        corpus = pyconll.load_from_file(str(conllu_file))
        
        processed_sentence_num = 0
        token_num = 0

        inclempty_count = 0
        cfcontained_count = 0
        contain_none_count = 0
        nonproj_count = 0
        root_nonproj_count = 0

        with open(os.path.join(output_dir, f'{conllu_file.stem}.txt'), 'w') as f, \
             open(os.path.join(output_dir,f'{conllu_file.stem}.tokens'), 'w') as g, \
             original_deptree_dir.joinpath(conllu_file.name).open('a') as h:
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
                    if args.write_deptree:
                        h.write(sentence.conll())
                        h.write('\n\n')

                    processed_sentence_num += 1
                    token_num += len(sentence)
                    # extract 5000 sentence for dev/test set
                    if "train" not in conllu_file.stem and processed_sentence_num == args.dev_test_sentence_num:
                        break
                    if "train" in conllu_file.stem and token_num > int(args.train_token_num):
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
                    logger.info(f'Cf contained in {sentence_to_str(sentence)}')
                    cfcontained_count += 1
                    continue

        logger.info(f'Corpus size (sent): {len(corpus)}')
        logger.info(f'Converted sentences: {processed_sentence_num}')
        logger.info(f'and tokens: {token_num}')

        logger.info(f'Non-projective sentences: {nonproj_count}')
        logger.info(f'Root-non-projective sentences: {root_nonproj_count}')
        logger.info(f'Sentences with None: {contain_none_count}')
        logger.info(f'Sentences with empty node: {inclempty_count}')
        logger.info(f'Sentences with control character: {cfcontained_count}')


if __name__ == '__main__':
    args = parser.parse_args()
    source_path, method_str, output_dir = generate_path_info(args)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    handler = FileHandler(filename=f'{output_dir}/convert.log')
    handler.setLevel(DEBUG)
    handler.setFormatter(Formatter(fmt))
    logger.addHandler(handler)

    convert_conllu_files(args)
