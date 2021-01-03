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
