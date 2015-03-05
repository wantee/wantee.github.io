---
layout: post
title: "RNNLIB: Softmax Layer"
date: 2015-02-05 21:08:17 +0800
comments: true
categories: [Neural Network]
---

* list element with functor item
{:toc}

I used to think that, in order to get the proper gradient, we have to take derivative of 
$\log$ of softmax with respect to weights. However,
the RNNLIB shows that we can actually factorize the network into single layers. In this post, 
we look into the Softmax Layer.

## Fundamentals

### List of Symbols

| Symbol     | Meaning     |
| ---------- |-------------|
| $J$        | cost function       |
| $y_k$      | activation of a neon  |
| $u_k$      | input of a neon  |  
| $S_i(\mathbf{u})$  | softmax function, $i$th value for a vector $\mathbf{u}$ |


### Formulas

Softmax of a vector $\mathbf{u}$ is defined as,

$$\begin{equation} \label{eq:softmax}
S_i(\mathbf{u}) = \frac{e^{u_i}}{\sum_k{e^{u_k}}} = y_i
\end{equation}$$


the derivative of softmax is,

$$\begin{equation} \label{eq:softmax_dev}
    \frac{\partial S_i(\mathbf{u})}{\partial u_j} =
    \frac{\partial y_i}{\partial u_j} = 
          \left\{\begin{array}{ll}
                        y_i(1-y_i) & i = j \\
                        -y_iy_j & i \neq j 
                \end{array} \right.
\end{equation}$$

### Layers in RNNLIB

Every layer in RNNLIB consists of input and output sides,
both sides contain activations and errors.
Their relations with terms in math are shown in following table,

| Variable              | Term          |
| --------------------- |:-------------:|
| *inputActivations*    | $u_k$         |
| *outputActivations*   | $y_k$         |
| *inputErrors*         | $\frac{\partial J}{\partial u_k}$  |  
| *outputErrors*        | $\frac{\partial J}{\partial y_k}$  |


## Forward Pass

Forward pass computes $y_k$ from $u_k$ using equation 
\eqref{eq:softmax}. There is a trick in the code, 
we can call it the *safe* softmax.

To understand it, consider dividing both numerator and denominator
by $e^c$ in equation \eqref{eq:softmax}, 
 
$$\begin{equation}
S_i(\mathbf{u}) 
= \frac{\frac{e^{u_i}}{e^{c}}}{\frac{\sum_k{e^{u_k}}}{e^c}} 
= \frac{e^{u_i - c}}{\sum_k{e^{u_k - c}}} 
= S_i(\hat{\mathbf{u}})  
\end{equation}$$ 

thus, in order to avoid overflow when calculating exponentials[^1], 
we can replace $u_k$ with $\hat{u}\_k=u\_k-c$. Typically, $c$ is set to $u\_{max}$.

In RNNLIB, $$c=\frac{u\_{max}+u\_{min}}{2}$$.

## Backpropagating

Backpropagation computes $\frac{\partial J}{\partial u_k}$ 
from $\frac{\partial J}{\partial y_k}$.

In RNNLIB, the result is

$$\begin{equation} \label{eq:error_u_res}
\frac{\partial J}{\partial u_j} = y_j (\frac{\partial J}{\partial y_j} 
- \langle \mathbf{y}, \frac{\partial J}{\partial \mathbf{y}} \rangle)
\end{equation}$$

where, $\langle \cdot \, , \cdot \rangle$ denotes inner product.

To get the above equation, we first notice that variations in 
$u_j$ give rise to variations in the error function $J$ 
through variations in all $y_k$s. 
Thus, according to the [Multivariable Chain Rules](https://www.math.hmc.edu/calculus/tutorials/multichainrule/),
we can write,

$$\begin{equation} \label{eq:error_u}
\frac{\partial J}{\partial u_j} = \sum_k{\frac{\partial J}{\partial y_k}\frac{\partial y_k}{\partial u_j}}
\end{equation}$$

Using equation \eqref{eq:softmax_dev} to replace $\frac{\partial y_k}{\partial u_j}$, we get,

$$\begin{equation} 
\begin{split}
\frac{\partial J}{\partial u_j} &= y_j(1-y_j)\frac{\partial J}{\partial y_j} +
\sum_{k: k\neq j}{-y_k y_j \frac{\partial J}{\partial y_k}} \\
&= y_j(\frac{\partial J}{\partial y_j} -y_j \frac{\partial J}{\partial y_j} 
+ \sum_{k: k\neq j}{-y_k \frac{\partial J}{\partial y_k}}) \\
&= y_j(\frac{\partial J}{\partial y_j} - \sum_{k}{y_k \frac{\partial J}{\partial y_k}}) \\
&= y_j(\frac{\partial J}{\partial y_j} - \langle \mathbf{y}, \frac{\partial J}{\partial \mathbf{y}} \rangle)
\end{split}
\end{equation}$$

Finally, we reach equation \eqref{eq:error_u_res}.

In this way, softmax operation can be implemented to be a standalone layer.

[^1]: Strictly speaking, this converts overflow into underflow. 
      Underflow is no problem, because that rounds off to zero, which is a well-behaved floating point number.
      otherwise, it will be Infinity or NaN. see [this article](http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/) for details.
