---
layout: post
title: "RNNLIB: Introduction"
date: 2015-02-05 16:02:28 +0800
comments: true
categories: [Neural Network]
---

RNNLIB is a recurrent neural network library for sequence learning problems,
which is written by [Alex Graves](http://www.cs.toronto.edu/~graves/).

In [this paper](http://www6.in.tum.de/pub/Main/Publications/Graves2006a.pdf),
Graves proposed the CTC(Connectionist Temporal Classification), 
which allows the system to transcribe unsegmented sequence data. 
The most exciting thing is that by training a deep bidirectional 
LSTM network with CTC, it is possible to 
perform automatic speech recognition in 
an [end-to-end fashion](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf), 
i.e. without any human expertise.

RNNLIB covers all the theories in Graves's paper, including:

* Bidirectional Long Short-Term Memory
* Connectionist Temporal Classification
* Multidimensional Recurrent Neural Networks

I will try to explain the codes in RNNLIB in following posts.

1. {% post_link 2015-02-05-rnnlib-softmax-layer %} 
2. {% post_link 2015-02-08-rnnlib-connectionist-temporal-classification-and-transcription-layer %}

