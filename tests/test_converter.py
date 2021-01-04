from src.converter import *
import pyconll

SAMPLE_CONLLU = """
1	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	2	nsubj	_	Discourse=sequence:112->24|Entity=(person-2)
2	heard	hear	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	_	_
3	a	a	DET	DT	Definite=Ind|PronType=Art	4	det	_	Entity=(abstract-138
4	noise	noise	NOUN	NN	Number=Sing	2	obj	_	_
5	like	like	ADP	IN	_	6	case	_	_
6	paper	paper	NOUN	NN	Number=Sing	4	nmod	_	Entity=(object-118)abstract-138)|SpaceAfter=No
7	.	.	PUNCT	.	_	2	punct	_	_
"""

sentence = pyconll.load.load_from_string(SAMPLE_CONLLU)[0]


def test_extract_children():
    assert extract_children(sentence, sentence[1]) == [1, 2, 4, 7]


def test_extract_left_children():
    assert extract_left_children(sentence, sentence[1]) == [1]


def test_extract_right_children():
    assert extract_right_children(sentence, sentence[1]) == [4, 7]


def test_flat_X_converter():
    assert flat_converter(sentence, sentence[1], get_X_nt).rstrip(
    ) == "(X (X I) (X heard) (X (X a) (X noise) (X (X like) (X paper))) (X .))"


def test_flat_pos_converter():
    assert flat_converter(sentence, sentence[1], get_pos_nt).rstrip(
    ) == "(VERBP (PRONP I) (VERBP heard) (NOUNP (DETP a) (NOUNP noise) (NOUNP (ADPP like) (NOUNP paper))) (PUNCTP .))"


def test_flat_merge_pos_converter():
    assert flat_converter(sentence, sentence[1], get_merge_pos_nt).rstrip(
    ) == "(VERBP (NOUNP I) (VERBP heard) (NOUNP (NOUNP a) (NOUNP noise) (NOUNP (ADPP like) (NOUNP paper))) (PUNCTP .))"


def test_flat_dep_converter():
    assert flat_converter(sentence, sentence[1], get_dep_nt).rstrip(
    ) == "(root (nsubj I) (root heard) (obj (det a) (obj noise) (nmod (case like) (nmod paper))) (punct .))"


"""
def test_left_converter():
    assert left_converter(
        sentence, sentence[1]
    ) == "(VERBP (PRONP I) (VERBP (VERBP (VERBP heard) (NOUNP (DETP a) (NOUNP (NOUNP noise) (NOUNP (ADPP like) (NOUNP paper))))) (PUNCTP .)))"


def test_right_converter():
    assert right_converter(
        sentence, sentence[1]
    ) == "(VERBP (VERBP (VERBP (PRONP I) (VERBP heard)) (NOUNP (NOUNP (DETP a) (NOUNP noise)) (NOUNP (ADPP like) (NOUNP paper)))) (PUNCTP .))"
"""