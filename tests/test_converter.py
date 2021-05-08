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

ROOT_NONPROJ = """
1	That	that	PRON	_	Number=Sing|PronType=Dem	3	nsubj	_	_
2	is	be	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	cop	_	_
3	why	why	ADV	_	PronType=Int	8	advmod	_	_
4	the	the	DET	_	Definite=Def|PronType=Art	5	det	_	_
5	crucifixes	crucifixe	NOUN	_	Number=Plur	8	nsubj:pass	_	_
6	must	must	AUX	_	VerbForm=Fin	8	aux	_	_
7	be	be	AUX	_	VerbForm=Inf	8	aux:pass	_	_
8	replaced	replace	VERB	_	Tense=Past|VerbForm=Part|Voice=Pass	0	root	_	_
9	by	by	ADP	_	_	11	case	_	_
10	the	the	DET	_	Definite=Def|PronType=Art	11	det	_	_
11	image	image	NOUN	_	Number=Sing	8	obl	_	_
12	of	of	ADP	_	_	14	case	_	_
13	our	we	PRON	_	Number=Plur|Person=1|Poss=Yes|PronType=Prs	14	nmod:poss	_	_
14	Messiah	Messiah	PROPN	_	Number=Sing	11	nmod	_	_
15	.	.	PUNCT	_	_	3	punct	_	_
16	"	"	PUNCT	_	_	3	punct	_	_
"""

sentence = pyconll.load.load_from_string(SAMPLE_CONLLU)[0]

nonproj_sentence = pyconll.load.load_from_string(ROOT_NONPROJ)[0]


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


def test_left_X_converter():
    assert left_converter(sentence, sentence[1], get_X_nt).rstrip(
    ) == "(X (X I) (X (X (X heard) (X (X a) (X (X noise) (X (X like) (X paper))))) (X .)))"


def test_left_pos_converter():
    assert left_converter(sentence, sentence[1], get_pos_nt).rstrip(
    ) == "(VERBP (PRONP I) (VERBP (VERBP (VERBP heard) (NOUNP (DETP a) (NOUNP (NOUNP noise) (NOUNP (ADPP like) (NOUNP paper))))) (PUNCTP .)))"


def test_left_merge_pos_converter():
    assert left_converter(sentence, sentence[1], get_merge_pos_nt).rstrip(
    ) == "(VERBP (NOUNP I) (VERBP (VERBP (VERBP heard) (NOUNP (NOUNP a) (NOUNP (NOUNP noise) (NOUNP (ADPP like) (NOUNP paper))))) (PUNCTP .)))"


def test_left_dep_converter():
    assert left_converter(sentence, sentence[1], get_dep_nt).rstrip(
    ) == "(root (nsubj I) (root (root (root heard) (obj (det a) (obj (obj noise) (nmod (case like) (nmod paper))))) (punct .)))"


def test_right_X_converter():
    assert right_converter(sentence, sentence[1], get_X_nt).rstrip(
    ) == "(X (X (X (X I) (X heard)) (X (X (X a) (X noise)) (X (X like) (X paper)))) (X .))"


def test_right_pos_converter():
    assert right_converter(sentence, sentence[1], get_pos_nt).rstrip(
    ) == "(VERBP (VERBP (VERBP (PRONP I) (VERBP heard)) (NOUNP (NOUNP (DETP a) (NOUNP noise)) (NOUNP (ADPP like) (NOUNP paper)))) (PUNCTP .))"


def test_right_merge_pos_converter():
    assert right_converter(sentence, sentence[1], get_merge_pos_nt).rstrip(
    ) == "(VERBP (VERBP (VERBP (NOUNP I) (VERBP heard)) (NOUNP (NOUNP (NOUNP a) (NOUNP noise)) (NOUNP (ADPP like) (NOUNP paper)))) (PUNCTP .))"


def test_right_dep_converter():
    assert right_converter(sentence, sentence[1], get_dep_nt).rstrip(
    ) == "(root (root (root (nsubj I) (root heard)) (obj (obj (det a) (obj noise)) (nmod (case like) (nmod paper)))) (punct .))"

def test_rootcross_included():
    assert not rootcross_included(sentence) and rootcross_included(nonproj_sentence)
