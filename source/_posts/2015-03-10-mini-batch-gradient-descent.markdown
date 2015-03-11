---
layout: post
title: "Mini-Batch Gradient Descent"
author: Wantee Wang
date: 2015-03-10 21:44:23 +0800
comments: true
categories: [ Gradient Descent ]
header-includes:
   - \usepackage{graphicx}
   - \usepackage[all]{hypcap}
---

In Mini-Batch Learning, we update the parameter $\mathbf{w}$ every $b$ examples. There are two ways to do the update.

First, using the summation of all examples in the mini-batch, i.e.,

$$\begin{equation}\label{eq:sum}
  \Delta\mathbf{w} = - \alpha_1 \sum_{i=l}^{l+b-1}{\nabla E^{(i)}}
\end{equation}$$

Second, using the average of all examples in the mini-batch, i.e.,

$$\begin{equation}\label{eq:avg}
  \Delta\mathbf{w} = - \alpha_2 \frac{1}{b} \sum_{i=l}^{l+b-1}{\nabla E^{(i)}}
\end{equation}$$

From \eqref{eq:sum} and \eqref{eq:avg}, we can see that by simply scaling the learning rate, i.e. $\alpha_1 = \frac{1}{b} \alpha_2$, these two method can be equivalent. 

{% comment %} 
However, if we using some other optimization techniques, these methods act differently. For example, if we use momentum, \eqref{eq:sum} and \eqref{eq:avg} become,

$$\begin{equation}\label{eq:sum_mom}
  \Delta\mathbf{w}(t) = - \alpha_1 \sum_{i=l}^{l+b-1}{\nabla E^{(i)}} + \beta_1{\Delta\mathbf{w}(t-1)}
\end{equation}$$

and

$$\begin{equation}\label{eq:avg_mom}
  \Delta\mathbf{w}(t) = - \alpha_2 \frac{1}{b} \sum_{i=l}^{l+b-1}{\nabla E^{(i)}} + \beta_2{\Delta\mathbf{w}(t-1)}
\end{equation}$$
{% endcomment %}
