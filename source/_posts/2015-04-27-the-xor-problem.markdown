---
layout: post
title: "The XOR Problem"
author: Wantee Wang
date: 2015-04-27 10:39:02 +0800
comments: true
categories: [Machine Learning]
header-includes:
   - \usepackage{graphicx}
   - \usepackage[all]{hypcap}
---

* list element with functor item
{:toc}

The XOR is an interesting problem, not only because it is a classical example for *Linear Separability*, but also it played a significant role in the history of neutral network research.
 
## Probelm

The truth table for XOR is

| x | y | x *xor* y |
| - | - | ----------- |
| 0 | 0 |      0      |
| 0 | 1 |      1      |
| 1 | 0 |      1      |
| 1 | 1 |      0      |


It is impossible for a classifier with linear decision boundary to learn an XOR function. This can be seen easily by the following plot{% comment %} FOR-LATEX (\autoref{fig:xor}) {% endcomment %}.

{% img center /images/posts/xor.png "The XOR Problem" "fig:xor" %}

Apparently, we can't using a line to separate the two classes.

## Non-linear Boundary

If we take a carefully look at the scatter figure, it can be found that it's easy to use an ellipse or hyperbola to separate the classes.

Recall that, the general equation for ellipse or hyperbola is

$$\begin{equation}\label{eq:ellipse}
Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
\end{equation}$$

Thus, we can just feed those above components($x, y, x^2, y^2, xy$) to a linear classifier, and see if the classes can be separated.

We use [scikit-learn](http://scikit-learn.org/) to perform the experiments. Following shows the code,

{% gist 1c74b8336494cb0e9c6d xor-5d.py %}

After running, we see the final decision boundary{% comment %} FOR-LATEX (\autoref{fig:xor-5d}) {% endcomment %},

{% img center /images/posts/xor-5d.png "Non-linear boundary" "fig:xor-5d" %}

## Removing Redundant Features

It is can be seen that in the final equation \eqref{eq:ellipse}, $A = D$ and $C = E$(it will be more clear if we use logistic regression to fit the data). Actually, for boolean features, the high-order polynomial features are useless, because $\forall n, x\_i^n = x\_i$. So we can only use the interaction features($x\_ix\_j$). This time we get the features from [PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures) class of scikit-learn. 

{% gist 1c74b8336494cb0e9c6d xor-3d.py %}

Again, we show the final decision boundary{% comment %} FOR-LATEX (\autoref{fig:xor-3d}) {% endcomment %},

{% img center /images/posts/xor-3d.png "Non-linear boundary using 3-d features" "fig:xor-3d" %}

By adding polynomial features to the model inputs, we are actually mapping the features to higher dimension. This is the SVM's job, here we just choose the features manually.
 
