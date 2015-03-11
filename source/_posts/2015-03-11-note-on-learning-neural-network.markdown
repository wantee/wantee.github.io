---
layout: post
title: "Note on Learning Neural Network"
author: Wantee Wang
date: 2015-03-11 16:54:37 +0800
comments: true
categories: [Neural Network]
header-includes:
   - \usepackage{graphicx}
   - \usepackage[all]{hypcap}
---

This is a paper about back-propagation algorithm for Neural Network.

The table of contents is

{% comment %} FOR-TOC {% endcomment %}
* 1 Preliminary
  * 1.1 Non-linear function
    * 1.1.1 Sigmoid
    * 1.1.2 Hyperbolic tangent
    * 1.1.3 Softmax
  * 1.2 Cross entropy
  * 1.3 Gradient descent
    * 1.3.1 Batch Gradient Descent
    * 1.3.2 Stochastic Gradient Descent
  * 1.4 The Multivariable Chain Rule
  * 1.5 Network architecture
* 2 Feed-forward Network
  * 2.1 Forward pass
  * 2.2 Backpropagation
    * 2.2.1 Weight between hidden layer and output layer
    * 2.2.2 Weight between input layer and hidden layer
* 3 Recurrent Neural Network
  * 3.1 Forward pass
  * 3.2 Backpropagation
    * 3.2.1 Real-Time Recurrent Learning
    * 3.2.2 Backpropagation Through Time
{% comment %} FOR-TOC-END {% endcomment %}

The pdf version is in [this link]({% comment %} FOR-PDFLINK {% endcomment %}http://wantee.github.io/assets/miscs/BP-0.3.pdf{% comment %} FOR-PDFLINK-END {% endcomment %}).

