---
layout: post
title: "Discriminative vs Generative"
author: Wantee Wang
date: 2015-03-22 22:44:20 +0800
comments: true
categories: [Machine Learning]
header-includes:
   - \usepackage{graphicx}
   - \usepackage[all]{hypcap}
---

Models in Machine Learning can often be divided into two main categories, *Generative* and *Discriminative*.
The fundamental difference between them is:

* Discriminative models learn the (hard or soft) boundary between classes
* Generative models model the distribution of individual classes

In mathematics, discriminative models directly estimate posterior probabilities $P(y\mathop{\|}x)$, while generative models model class-conditional pdfs $p(x\mathop{\|}y)$ and prior probabilities $P(y)$, therefore the joint probability distributions $p(x,y)$.

Generative models often make some assumption on the underlying probability distributions and model it. Thus it is can be used to generate new samples from the learned distribution.

A simple way to distinct the two models is by considered the examples used during training. Generative model only needs examples of a particular class which it modelling. However, Discriminative model needs examples of at least two classes to find the boundary. 

## Examples

Some models can be seen as generative-discriminative pairs, e.g.,

* Classifiers: Naive Bayes and Logistic Regression
* Sequential Data: HMM and CRF

Neutral networks are discriminative model because they compute $p(output\mathop{\|}input)$.

## Discriminative and Generative Training

Training approaches can also be classified as discriminative or generative. Even though with the same model, we can choose different training approaches.

For example, the HMM-GMM model used in speech recognition, when we do MLE training with Baum–Welch algorithm, we are using a generative training method. However when we do MPE training, we are using a discriminative training method.


