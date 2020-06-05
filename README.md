# Parameter-free Sentence Embedding via Orthogonal Basis

This repo contains the official implementation for the EMNLP 2019 paper (oral presentation) 
[Parameter-free Sentence Embedding via Orthogonal Basis](https://arxiv.org/pdf/1810.00438.pdf), 

by __Ziyi Yang__ . Stanford University.

-------------------------------------------------------------------------------------
We propose a simple and robust non-parameterized approach for building sentence representations. Inspired by the Gram-Schmidt Process in geometric theory, we build an orthogonal basis of the subspace spanned by a word and its surrounding context in a sentence. We model the semantic meaning of a word in a sentence based on two aspects. One is its relatedness to the word vector subspace already spanned by its contextual words. The other is the word's novel semantic meaning which shall be introduced as a new basis vector perpendicular to this existing subspace. Following this motivation, we develop an innovative method based on orthogonal basis to combine pre-trained word embeddings into sentence representations. This approach requires zero parameters, along with efficient inference performance. We evaluate our approach on 11 downstream NLP tasks. Our model shows superior performance compared with non-parameterized alternatives and it is competitive to other approaches relying on either large amounts of labelled data or prolonged training time.

## Dependencies

* Python 3.7

* numpy

* scipy

## Instructions for Running GEM on STS benchmarks

1. First download word vectors [here](https://drive.google.com/drive/folders/1FB5xJ1O8zZ8PiKygp0J7P9mShHv5AunI?usp=sharing). Unzip and rename it to data.

2. To test on STS benchmarks dev, python code/encoder.py
