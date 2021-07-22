# Dep-to-Const Converter

This repository provides the converter from dependency structure to phrase structure.
The converter works completely algorithmically and does not need any training.

## Usage
### Requirement
`pip install pyconll nltk`

For test, you additionally need to install `pytest`.

### How to run
You can run the converter in the following format.

```
python converter.py
  --source_path <conll-formatted texts>
  --output_source_dir <output source directory>
  --convert_method <flat/left/right>
```

If source_path includes the treebank named train/dev/test set, the converted treebank is named according to the original name such as `train.txt`.
In addition to converted treebank, the program outputs raw text separated by space, which is named such as `train.tokens`.

Besides above, you need to set the labeling method (described later).
Currently we have four types of labeling methods and you need to choose one from them:

```
--without_label
--use_pos_label
--use_merged_pos_label
--use_dep_label
```

Example: Converting a Universal Dependencies corpus via flat conversion and without_label method.

```
python converter.py
  --source_path path/to/ud-treebanks-v2.7/English_EWT
  --output_source_dir path/to/output_dir
  --convert_method flat
  --without_label
```

## Description
The converter takes two types of parameter: conversion method and labeling policy.

### Conversion method
Three types of methods are implemented: flat/left/right.
These conversion algorithms work for any dependency structures, so long as they are projective.
The details of algorithm is described in [Collins et al, 1999](https://www.aclweb.org/anthology/P99-1065.pdf).

In short, flat method converts the dependency tree into the "flat" constituency tree.
It converts dependents into children in the constituency tree at the same time.

Left/right methods does not convert at the same time. Instead, they merge head and dependent one by one, in left-first or right-first way.
These methods produce binary tree.

### Labeling policy
We set following four types of nonterminals attached to resulting constituency tree:

- Without label: All the nonterminals are set 'X'
- POS label: Using part-of-speech tag of the head token.
- Merged-POS label: Basically using POS label, and convert DETP/PROPNP/PRONP into NOUNP.
- DEP label: Using dependency label of the head token.
