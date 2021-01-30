# Dep-to-Const Converter

This repository provides the converter from dependency structure to phrase structure.
The converter works completely algorithmic and does not need any training.

## Usage
### Requirement
`pip install pyconll nktk`

For test, you additionally need to install `pytest`.

### How to run
You can run the converter in the following format.

```
python converter.py
  --source_path <conll-formatted texts>
  --output_source_dir <output source directory>
  --convert_method <flat/left/right>
```

Source_path should include either train/dev/test set. The converted treebank is named according to the original name.

In addition to above, you need to set the labeling policy (described later).
Currently we have four types of labeling policies and you need to choose one from them:

```
--without_label
--use_pos_label
--use_merged_pos_label
--use_dep_label
```

Example: Converting a Universal Dependencies corpus via flat method and without_label policy.

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

In short, flat method converts dependency structure into "flat" constituency structure.
It converts dependents into children in the constituency structure at the same time.

Left/right methods does not convert at the same time. Instead, they merge head and dependeny one by one, in left-first or right-first way.
These methods produce complete binary tree.
As the algorithm suggests, constituency tree converted by left-first method tends to be right-branching (and vice verca for right-first method).

### Labeling policy
We set following four types of nonterminals attatched to resulting constituency tree:

- Without label: All the nonterminals are set 'X'
- POS label: Using part-of-speech tag of the head token.
- Merged-POS label: Basically using POS label, and convert DETP/PROPNP/PRONP into NOUNP.
- DEP label: Using dependency lagel of the head token.
