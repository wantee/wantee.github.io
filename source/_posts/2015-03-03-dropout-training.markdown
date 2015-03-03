---
layout: post
title: "Dropout Training"
date: 2015-03-03 15:18:26 +0800
comments: true
categories: [Neural Network]
---

* list element with functor item
{:toc}

Dropout is a regularisation technique for reducing over-fitting in large neural nets. Hinton proposes the method in [this paper](http://arxiv.org/abs/1207.0580). 
Most materials are from [Srivastava's page](http://www.cs.toronto.edu/~nitish/dropout/).
 
It prevents overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently. The term “dropout” refers to dropping out units (hidden and visible) in a neural network.

## The Method

There are 2 key points for dropout learning: 

* a) Dropping units while training; 
* b) Scaling weight while testing. 

As shown in following figure, where $p$ is the dropout retention rate.

{% img center /images/posts/Dropout.png Dropout %}

Units to be dropped is chosen in a random way. Note that dropping a unit out means temporarily removing it from the network, along with all its incoming and outgoing connections. Therefore we have to deal with it both during forward pass and backpropagation.

Applying dropout to a neural network amounts to sampling a “thinned” network from it. A neural net with $n$ units, can be seen as a collection of $2^n$ possible thinned neural networks. For each presentation of each training case, a new thinned network is sampled and trained. 

At test time, the ideal way is to explicitly average the predictions from exponentially many thinned models, which is obviously not feasible. However the network with scaled weights gives a good approximation. 

The goal is that for any hidden unit the expected output (under the distribution used to drop units at training time) is the same as the actual output at test time.

The expected output of a unit is 

$$
\mathbb{E}[\mathbf{y}_j] = \sum_i{p \mathbf{w}_{ji} \mathbf{x}_i}
$$

Therefore, by scaling down the weight used at test time, i.e. $\mathbf{w}'\_{ji} = p\mathbf{w}\_{ji}$, we can achieve the goal. This is the way used in the above paper and shown in the figure.

Another way is to scale up the output at training time to the same magnitude as test time, i.e. $\mathbf{y}'\_{j} = \frac{1}{p}\mathbf{y}\_j$.

## Implementation in Kaldi

Both Karel's and Dan's implementation have the Dropout codes, with some differences.

Karel's code(`src/nnet/nnet-activation.h:Dorpout`) uses the scale-up method to get the expected output. Dropping out is implemented during forward pass and by storing the dropped out units using a 0/1 vector, the back-propagated derivative can be set properly.

Dan's code(`src/nnet2/net-component.cc:DropoutComponent`) use a clever way to avoid storing the dropping units. While backpropagation, we can get the input error $\mathbf{e\_i}$ from output error $\mathbf{e\_i}$ by

$$
\mathbf{e_i} = \frac{\mathbf{a_o}}{\mathbf{a_i}} \mathbf{e_o}
$$

where $\mathbf{a\_i}$ and $\mathbf{a\_o}$ is the activation of input and output for Dropout component. Elements in $\mathbf{a\_o}$ is the equal to that in $\mathbf{a\_i}$ except the dropping ones, which is zero.

Dan's code seems to not care about the scaling problem, we can scale the final network before testing using the scale-down method. However, there is another form of scale in Dan's code, instead of set the output of dropping unit to zero, we just scale the output value by a factor $\alpha$. To get a proper scaled version of output, we'd like to scale all the units besides the dropping ones and make it satisfy that the expected scale factor should be 1, i.e.,

$$
p \alpha + (1-p)\beta = 1
$$

Therefore, we can get the factor of other units $\beta = \frac{1-p\alpha}{1-p}$.
