---
layout: post
title: "RNNLIB: Connectionist Temporal Classification and Transcription Layer"
date: 2015-02-08 16:56:40 +0800
comments: true
categories: [Neural Network]
---

* list element with functor item
{:toc}

CTC is the core concept make it possible to transcribe unsegmented sequence data.
RNNLIB implements it in a single layer called Transcription Layer.
We go into this particular layer in this post, the main reference is the Graves'
[original paper](http://www6.in.tum.de/pub/Main/Publications/Graves2006a.pdf).

The key point for CTC is to use a simple map transforming the RNN output to unsegmented labelling,
and construct a new objective function based on the map.
This map do not need a precise alignment, thus greatly simplify the task and reduce human expert involvement. 

## The Name

"Connectionist" is the adjective form of "connectionism", 
[Connectionism](http://en.wikipedia.org/wiki/Connectionism) is a terminology in cognitive science,
which models mental or behavioural phenomena as the emergent processes of interconnected networks of simple units.
The most common forms use neural network models. 

In the traditional neural network recipe, we independently model the input sequence 
in each time-step or frame. This can be referred as *framewise classification*.
[Kadous](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.8007&rep=rep1&type=pdf) 
extends the classification paradigm to multivariate time series, and 
names it as *temporal classification*.
With this, we do not have to label every time step in training data set.

Combining RNN and temporal classification, Graves proposes the *connectionist temporal classification*.

To distinguish from classification, RNNLIB implements the CTC as *Transcription Layer*, 
indicating that with CTC we can directly transcribe input sequence(e.g. acoustic signal)
into output sequence(e.g. words).

## The Theory

### List of Symbols

Following the notations in the paper, we first list the symbols.

| Symbol     | Meaning     |
| ---------- |-------------|
| $L$  | (finite) alphabet of labels  |
| $L'$  | $L \cup \\{blank\\}$  |
| $\mathcal{X}$  | $(\mathbb{R}^m)^{*}$, $m$ dimensional input space  |  
| $\mathcal{Z}$  | $L^{*}$, output space, set of all sequences over the $L$ |
| $\mathcal{D_{X \times Z}}$      | underlying distribution of data  |
| $S$        | set of training examples supposed to be drawn from $\mathcal{D\_{X \times Z}}$     |
| ($\mathbf{x},\mathbf{z})$        | example in $S$, $\mathbf{x} = (x\_1, x\_2, \dotsc, x\_T)$, $\mathbf{z} = (z\_1, z\_2, \dotsc, z\_U)$ and $U \leq T$     |
| $h:\mathcal{X} \mapsto \mathcal{Z}$ | temporal classifier to be trained |
| $\mathcal{N}\_{w}:(R^{m})^{T} \mapsto (R^n)^{T}$ | RNN, with $m$ inputs, $n$ outputs and weight vector $w$, as a continuous map | 
| $\mathbf{y} = \mathcal{N}\_{w}$ | sequence of RNN output |
| $y\_{k}^{t}$ | the activation of output unit $k$ at time $t$ |
| $\pi$ | *path*, element of $L'^{T}$ |
| $\mathbf{l} \in L^{\leq T}$ | label sequence or *labelling* |
| $\mathcal{B}:L'^{T} \mapsto L^{\leq T}$ | map from path to labelling |
| $\mathbf{l}\_{a\mathord{:}b}$ | sub-sequence of $\mathbf{l}$ from $a$th to $b$th labels |
| $\mathbf{l}'$ | modified label sequence, with blanks added to the beginning and the end and inserted between every pair of labels in $\mathbf{l}$ |
| $\alpha\_t(s)$ | forward variable, the total probability of $\mathbf{l}\_{1:s}$ at time $t$ |
| $\beta\_t(s) $ | backward variable, the total probability of $\mathbf{l}\_{s:\|\mathbf{l}'\|}$ at time $t$ |
| $\tilde{\beta}\_t(s) $ | backward variable, the total probability of $\mathbf{l}\_{s:\|\mathbf{l}'\|}$ start at time $t+1$ |
| $O^{ML}(S,\mathcal{N}\_{w})$ | maximum likelihood objective function |
| $\delta\_{kk'}$ | [Kronecker delta](http://en.wikipedia.org/wiki/Kronecker_delta) |


### Training Procedure

The goal is to use $S$ to train a temporal classifier $h$ to classify previously unseen input sequences in a way that minimises the ML objective function:

$$\begin{equation} \label{eq:obj_ml}
O^{ML}(S,\mathcal{N}_{w}) = - \sum_{(\mathbf{x},\mathbf{z})\in S}{\ln(p(\mathbf{z}|\mathbf{x}))}
\end{equation}$$

To train the network with gradient descent, 
we need to differentiate \eqref{eq:obj_ml} with respect to the network outputs. 
Since the training examples are independent we can consider them separately:

$$\begin{equation} \label{eq:obj}
\frac{\partial O^{ML}(\{(\mathbf{x},\mathbf{z}\},\mathcal{N}_{w})}{\partial y_k^t} 
    = - \frac{\partial \ln(p(\mathbf{z}|\mathbf{x}))}{\partial y_k^t}
    = - \frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{\partial p(\mathbf{z}|\mathbf{x})}{\partial y^t_k}
\end{equation}$$

Another thing we have to consider is how to map from network outputs to labellings.
Use $\mathcal{B}$ to denote such a map. Given a path, we simply removing all blanks 
and repeated labels and the remaining labels form a labelling(e.g. $\mathcal{B}(a-ab-)=\mathcal{B}(-aa--abb)=aab$). 

Then we can define the conditional probability of a labelling,

$$\begin{equation} \label{eq:labelling}
p(\mathbf{l}|\mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})}{p(\pi|\mathbf{x})}
\end{equation}$$

where, $p(\pi\|\mathbf{x})$ is the conditional probability of a path given $\mathbf{x}$, and is defined as:

$$\begin{equation} \label{eq:path}
p(\pi|\mathbf{x}) = \prod_{t=1}^{T}{y_{\pi_t}^{t}},\forall \pi \in L'^{T}
\end{equation}$$

To calculate \eqref{eq:obj}, we first define the forward and backward variable,

$$\begin{equation} \label{eq:fwd}
\alpha_t(s) = \sum_{\pi \in L^{T}:\mathcal{B}(\pi_{1\mathord{:}t})=\mathbf{l}_{1\mathord{:}s}}
   {\prod_{t'=1}^{t}{y^{t'}_{\pi_{t'}}}}
\end{equation}$$

$$\begin{equation} \label{eq:bwd}
\beta_t(s) = \sum_{\pi \in L^{T}:\mathcal{B}(\pi_{t\mathord{:}T})=\mathbf{l}_{s\mathord{:} |\mathbf{l}'|}}
   {\prod_{t'=t}^{T}{y^{t'}_{\pi_{t'}}}}
\end{equation}$$

Note that the product of the forward and backward variables at a given $s$ and $t$ is the probability of all the paths corresponding to $\mathbf{l}$ that go through the symbol $s$ at time $t$, i.e.,

$$\begin{equation} \label{eq:fwd_bwd_ori}
\alpha_t(s)\beta_t(s) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l}):\pi_{t}=\mathbf{l}'_{s}}
   {y^{t}_{\mathbf{l}'_{s}}\prod_{t=1}^{T}{y^{t}_{\pi_{t}}}}
\end{equation}$$

Rearranging and substituting in from \eqref{eq:path} gives,

$$\begin{equation} \label{eq:fwd_bwd}
\frac{\alpha_t(s)\beta_t(s)}{y^{t}_{\mathbf{l}'_{s}}} = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l}):\pi_{t}=\mathbf{l}'_{s}}
   {p(\pi|\mathbf{x})}
\end{equation}$$

For any $t$, we can therefore sum over all $s$ to get $p(\mathbf{l} \| \mathbf{x})$:

$$\begin{equation} \label{eq:labelling_fwd_bwd}
p(\mathbf{l}|\mathbf{x}) = \sum_{s=1}^{|\mathbf{l}'|}\frac{\alpha_t(s)\beta_t(s)}{y^{t}_{\mathbf{l}'_{s}}}
\end{equation}$$

On the other hand, combining \eqref{eq:labelling} and \eqref{eq:path},

$$\begin{equation} \label{eq:labelling_all}
p(\mathbf{l}|\mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})}{\prod_{t=1}^{T}{y_{\pi_t}^{t}}}
\end{equation}$$

Thus to differentiate this with respect to $y_k^t$ , 
we need only consider those paths going through label $k$ at time $t$
(derivatives of other paths is zero). 
Noting that the same label (or blank) may be repeated several times for a single labelling $\mathbf{l}$, 
we define the set of positions where label $k$ occurs as $lab(\mathbf{l},k) = \\{s : \mathbf{l}'_s = k\\}$, 
which may be empty. We then differentiate \eqref{eq:labelling_all} to get,

$$\begin{equation} \label{eq:labelling_drv}
\begin{split} 
\frac{\partial p(\mathbf{l}|\mathbf{x})}{\partial y_k^t} 
  &= \sum_{s \in lab(\mathbf{l}, k)}{\frac{\partial p(\pi|\mathbf{x})}{\partial y_k^t}} \\
  &= \sum_{s \in lab(\mathbf{l}, k)}{\frac{\partial \prod_{t'=1}^{T}{y_{\pi_{t'}}^{t'}}}{\partial y_k^t}} \\
  &= \sum_{s \in lab(\mathbf{l}, k)}{\frac{\partial y_k^t\prod_{t' \neq t}{y_{\pi_{t'}}^{t'}}}{\partial y_k^t}} \\
  &= \sum_{s \in lab(\mathbf{l}, k)}{\prod_{t' \neq t}{y_{\pi_{t'}}^{t'}}} \\
  &= \frac{1}{ {y^t_k}^2 } \sum_{s \in lab(\mathbf{l}, k)}{\alpha_t(s)\beta_t(s)}
\end{split} 
\end{equation}$$

At this point, we can set $\mathbf{l} = \mathbf{z}$ and substituting into \eqref{eq:obj}, then get the final gradient,

$$\begin{equation} \label{eq:grad}
\frac{\partial O^{ML}(\{(\mathbf{x},\mathbf{z}\},\mathcal{N}_{w})}{\partial y_k^t} 
    = - \frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{1}{ {y^t_k}^2 } \sum_{s \in lab(\mathbf{l}, k)}{\alpha_t(s)\beta_t(s)}
\end{equation}$$

where, $p(\mathbf{z}\|\mathbf{x})$ can be calculated from \eqref{eq:labelling_fwd_bwd}.

Next, we can give the gradient for the unnormalised output $u_k^t$. Recall that derivative of softmax function is, 

$$\begin{equation} \label{eq:err_softmax}
\frac{\partial y^t_{k'}}{\partial u_k^t} = y^t_{k'}\delta_{kk'} - y^t_{k'}y^t_k
\end{equation}$$

Then we get,

$$\begin{equation} \label{eq:error_u}
\begin{split} 
\frac{\partial O}{\partial u_k^t} 
  &= \sum_{k'}{\frac{\partial O}{\partial y^t_{k'}}\frac{\partial y^t_{k'}}{\partial u^t_k}} \\
  &= \sum_{k'}{((y^t_{k'}\delta_{kk'} - y^t_{k'}y^t_k) (-\frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{1}{ {y^t_{k'}}^2 } 
                \sum_{s \in lab(\mathbf{l}, k')}{\alpha_t(s)\beta_t(s)}))} \\
  &= \sum_{k'}{(\frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{y^t_{k}}{ {y^t_{k'}} } 
                \sum_{s \in lab(\mathbf{l}, k')}{\alpha_t(s)\beta_t(s)})} - \frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{1}{ {y^t_k} } 
                \sum_{s \in lab(\mathbf{l}, k)}{\alpha_t(s)\beta_t(s)}\\  
  &= y^t_k - \frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{1}{ {y^t_k} } 
                \sum_{s \in lab(\mathbf{l}, k)}{\alpha_t(s)\beta_t(s)}\\                               
\end{split} 
\end{equation}$$

we write the last step by noting that $\sum\_{k'}\sum\_{s \in lab(\mathbf{l}, k')}{(\cdot)} \equiv \sum\_{s=1}^{\|\mathbf{l}'\|}{(\cdot)}$,
then, using \eqref{eq:labelling_fwd_bwd}, the $p(\mathbf{z}\|\mathbf{x})$ is canceled out.

### The CTC Forward-Backward Algorithm

The last thing we have to do is calculating the forward and backward variables. We now show that by define a recursive from, 
these variables can be calculated efficiently.

Given a labelling $\mathbf{l}$, we first extend it to $\mathbf{l}'$ with blanks added to the beginning 
and the end and inserted between every pair of labels. 
The length of $\mathbf{l}'$ is therefore $2\|\mathbf{l}\| + 1$. 
In calculating the probabilities of prefixes of $\mathbf{l}'$ we allow all transitions between blank and non-blank labels, 
and also those between any pair of distinct non-blank labels(because of the map $\mathcal{B}$, the repeated labels will be merged). 
We allow all prefixes to start with either a blank ($b$) or the first symbol in $\mathbf{l}$ ($\mathbf{l}_1$).

This gives us the following rules for initialisation

$$
\begin{split} 
\alpha_1(1) &= y_b^1 \\ 
\alpha_1(2) &= y_{\mathbf{l}_1}^1 \\ 
\alpha_1(s) &= 0, \forall s > 2 \\                               
\end{split} 
$$

and recursion

$$ \begin{equation} \label{eq:alpha}
    \alpha_t(s) = \left\{\begin{array}{ll}
                                    y_{\mathbf{l}'_s}^t(\alpha_{t-1}(s) + \alpha_{t-1}(s-1)) & \mathbf{l}'_s = b\, \text{or} \, \mathbf{l}'_{s-2} = \mathbf{l}'_s\\
                                    y_{\mathbf{l}'_s}^t(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)) & otherwise
                                \end{array} \right.
\end{equation}$$

Note that $\alpha_t(s) = 0, \forall s < \|\mathbf{l}'\|−2(T −t)−1$, 
because these variables correspond to states for which there are not enough time-steps left to complete the sequence.

Here we can get another method to calculate $p(\mathbf{l} \| \mathbf{x})$, by adding up all forward variables at time $T$, i.e.,

$$ \begin{equation} \label{eq:porb}
    p(\mathbf{l} | \mathbf{x}) = \alpha_T(|\mathbf{l}'|) + \alpha_T(|\mathbf{l}'| - 1)
\end{equation}$$

Similarly, the backward variables can be initalisd as,

$$
\begin{split} 
\beta_T(|\mathbf{l}'|) &= y_b^T \\ 
\beta_T(|\mathbf{l}'| - 1) &= y_{\mathbf{l}_{|\mathbf{l}|}}^T \\ 
\beta_T(s) &= 0, \forall s< |\mathbf{l}'| - 1\\                               
\end{split} 
$$

and recursion

$$ \begin{equation} \label{eq:beta}
    \beta_t(s) = \left\{\begin{array}{ll}
                y_{\mathbf{l}'_s}^t(\beta_{t+1}(s) + \beta_{t+1}(s+1)) & \mathbf{l}'_s = b\, \text{or} \, \mathbf{l}'_{s+2} = \mathbf{l}'_s\\
                y_{\mathbf{l}'_s}^t(\beta_{t+1}(s) + \beta_{t+1}(s+1) + \beta_{t+1}(s+2)) & otherwise
                  \end{array} \right.
\end{equation}$$

Note that $\beta\_t(s) = 0, \forall s > 2t$.

Following figure illustrate the forward backward algorithm applied to the labelling 'CAT'(from the paper).
 
{% img center /images/posts/CTC-alpha-beta.png Alpha-Beta Algorithm %}

## The Implementation

The `TranscriptionLayer` class inherits the `SoftmaxLayer` class(see {% post_link 2015-02-05-rnnlib-softmax-layer this post %}).
The `feed_forward()` and `feed_back()` methods are the general softmax function, 
so only need to implement the `calculate_errors()` method to calculate the $\frac{\partial O}{\partial y\_k^t}$.
In order to use \eqref{eq:grad} to get output error, first need to calculate the $\alpha$s and $\beta$s.
Forward variables are got using \eqref{eq:alpha}. 

But backward variables are in another form, given in Graves' [Dissertation](www6.in.tum.de/Main/Publications/Graves2008c.pdf).
Consider backward variable started from time $t+1$,

$$\begin{equation} \label{eq:bwd_new}
\tilde\beta_t(s) = \sum_{\pi \in L^{T}:\mathcal{B}(\pi_{t\mathord{:}T})=\mathbf{l}_{s\mathord{:} |\mathbf{l}'|}}
   {\prod_{t'=t+1}^{T}{y^{t'}_{\pi_{t'}}}}
\end{equation}$$

Noting that, $\beta$ and $\tilde\beta$ has a simple relationship:

$$\begin{equation} \label{eq:bwd_relaion}
\beta_t(s) = y_{\pi_{t}}^t\tilde\beta_t(s)
\end{equation}$$

Thus, we can get recursion formula for $\tilde\beta$ by substituting \eqref{eq:bwd_relaion} into \eqref{eq:beta},

$$
\begin{split} 
\tilde\beta_T(|\mathbf{l}'|) &= 1 \\ 
\tilde\beta_T(|\mathbf{l}'| - 1) &= 1 \\ 
\tilde\beta_T(s) &= 0, \forall s< |\mathbf{l}'| - 1\\                               
\end{split} 
$$

$$ \begin{equation} \label{eq:beta_new}
    \tilde\beta_t(s) = \left\{\begin{array}{ll}
                y_{\mathbf{l}'_s}^{t+1}\tilde\beta_{t+1}(s) + y_{\mathbf{l}'_{s+1}}^{t+1}\tilde\beta_{t+1}(s+1) & \mathbf{l}'_s = b\, \text{or} \, \mathbf{l}'_{s+2} = \mathbf{l}'_s\\
                y_{\mathbf{l}'_s}^{t+1}\tilde\beta_{t+1}(s) + y_{\mathbf{l}'_{s+1}}^{t+1}\tilde\beta_{t+1}(s+1) + y_{\mathbf{l}'_{s+2}}^{t+1}\tilde\beta_{t+1}(s+2) & otherwise
                \end{array} \right.
\end{equation}$$

Noting that, if $\mathbf{l}'\_s \neq blank$, then $\mathbf{l}'\_{s+1}$ must be $blank$.

And the gradient for output \eqref{eq:grad} becomes,

$$\begin{equation} \label{eq:grad_new}
\frac{\partial O^{ML}(\{(\mathbf{x},\mathbf{z}\},\mathcal{N}_{w})}{\partial y_k^t} 
    = - \frac{1}{p(\mathbf{z}|\mathbf{x})} \frac{1}{ y^t_k } \sum_{s \in lab(\mathbf{l}, k)}{\alpha_t(s)\tilde\beta_t(s)}
\end{equation}$$

where,

$$\begin{equation}
p(\mathbf{z}|\mathbf{x}) = \sum_{s=1}^{|\mathbf{z}'|}{\alpha_t(s)\tilde\beta_t(s)}
\end{equation}$$

Actually, the RNNLIB code computes $p(\mathbf{z}\|\mathbf{x})$ using \eqref{eq:porb}.

To wrap up, CTC using a forward-backward algorithm to efficiently compute the RNN output errors, 
corresponding to a new ML objective function. With these errors, 
we can use any traditional gradient methods to train the network.
 
