# Attention Mechanism for Seq2Seq Models


1. TOC
{:toc}

<!--
## Goals

  1. what is attention
  2. Why attention
  3. Why is it better than RNN
  4. Is attention the go-to architecture for seq2seq models
  5. What attention can't do
  6. Where, when, how
  7. What are "long sequences" in general? 10 elements? where does it break exactly (for rnn and for attention)? How does this threshold depend on the largest sequence length in training data? 
  8. Is attention really necessary for short sequences? 
  9. Does it solve larger sequence length issues at prod? -->
  
  
## Prerequisites
You should be able to understand [this image](https://miro.medium.com/max/1400/1*Ismhi-muID5ooWf3ZIQFFg.png) or [this image](https://i.stack.imgur.com/f6DQb.png) or [this image](https://www.guru99.com/images/1/111318_0848_seq2seqSequ4.png).



## Introduction

In this blog, we will review the original attention mechanism as published in [Bahdanau et. al.](https://arxiv.org/pdf/1409.0473.pdf).

<!--and we will also give another example of attention known as "self-attention".-->

Let's first start with some background on where attention mechanism is mainly used -- seq2seq models. One classic example of the seq2seq model is Neural Machine Translation (NMT). Let's review it.

### Neural Machine Translation
NMT has two components:

1. Encoder: The encoder, usually an RNN or multi-layered RNNs, is used to encode the source language sentence into a fixed-length vector, which is the last hidden state of the RNN for a given input sequence. 

2. Decoder: The decoder takes in the last encoder hidden state as input and tries to predict each word in the target language (in an autoregressive way, meaning the last predicted word is used to predict the current word.)

The bottleneck with such a system is that the decoder has to predict the _entire_ sentence in the target language with only the last-encoder-hidden-state as input. This limits the capacity of the decoder to predict/output long sentences in the target language.

What if there was a way to supplement the last-encoder-hidden-state with some more information to make the task of decoding a little bit easier? 

**Enter attention!!**



## Learning to Align and Translate : The Attention Model

We have seen in the previous section that decoding the whole output sequence from a single fixed-length vector can be problematic when the output sequence is long. We can provide more info/context about all the encoder hiddens states by taking dynamically weighted average of all the encoder hidden states, in the following manner, known as "attention":

Assume that, $(h_{1}, h_{2}, h_{3},..., h_{T})$ are the hidden states of the encoder layer.

Let the decoder layer be a generic function, 

\begin{equation} 
y_{i} = f(y_{i-1}, c_i, s_i)
\end{equation} 

where 
  
  1. $y_{i-1}$ is the last predicted output by the decoder,
  2. $s_i$ is the current hidden state of the decoder,
  3. $c_i$ is the dynamic context vector.
  

The dynamic context vector $c_i$ is a (learnable) weighted average of all the encoder hidden states as described below:

\begin{equation} 
c_i = \sum_{j=1}^{T_{x}}\alpha_{ij}h_j.
\end{equation}

where $\alpha_{ij}$ is the weight/score of the hidden state $h_j$ and are given by equation:

\begin{equation} 
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}
\end{equation}

where
\begin{equation} 
e_{ij} = a(s_{i-1}, h_j)
\end{equation} 

The function $a$ here is an "attention" model. 

One can think of $e_{ij}$ as the unnormalized weight/score given to the encoder hidden state $h_j$ while calculating the linear combination of all hidden states to obtain a context vector $c_i$. The context vector is further used to obtain the decoder output at $i^{th}$ step. 

So, $a$ is a function (of $s_{i-1}$ and $h_j$) which learns how much weightage should be assigned to encoder hidden state $h_j$ when decoding $y_i$.



<!-- ## Attention Layer

Let's take a closer look at the (self) attention/alignment model in the below diagram:

![attention mechanism](/images/attention_scaled_down.jpg)
*Figure 1: Self-attention mechanism to calculate context vector for the first sequence element.*

1. We have a sequence of inputs $(x_1, x_2, x_3, ..., x_n)$ and an initial hidden state $h_0$ that are consumed by the RNN model one-by-one producing corresponding hidden states $(h_1, h_2, h_3, ..., h_n)$. This is shown in the bottom part of the image.
2. Let's calculate the weights $a_{ij}$ needed to calculate the context vector. This is shown in the middle part of the image.
    1. Let $f$ be some function s.t. $e_{ij} = f(h_i, h_j)$. Exact form/expression of $f$ is not necessary here, it can be anything, we just need a scalar out of $f$.
    2. We get $e_{11}, e_{12}, e_{13}, ..., e_{1n}$ from  $f(h_1, h_1), f(h_1, h_2), f(h_1, h_3), ..., f(h_1, h_n)$ respectively.
    3. This $e_{ij}$ is an unnormalized score/weight of $h_i$ and $h_j$. Take softmax of $e_{ij}$ for $j \in [1, n]$ to get $a_{ij}$, normalized score/weight : $a_{ij} = \frac{e_{ij}}{\sum_{j=1}^{T_{n}}e_{ij}}$. This will determine how much relative information from $h_j$ to include when calculating $c_i$.
3. Now finally, with $a_{ij}$ as weights, we take weighted average of the hidden states $(h_1, h_2, h_3, ..., h_n)$ to obtain the context vector, $c_i = \sum_{j=1}^{T_n} a_{ij}h_j$. This is in some sense an _expectation_ of hidden states $h_j$. This is depicted in the top part of the image under the "Addition" step.


This is a generic attention layer that can be put between any two layers where input is in the form of a sequence.

Now, how does it apply to seq2seq models, MT, in particular?

For MT, one little change reproduces the attention model mentioned in the above paper. In the above figure, the blue $h1$ is an encoder hidden state, hence $c1$ self-attention context vector. If instead of an encoder hidden state, we had the _decoder_ hidden state $s1$, then we would get the same attention model described in the paper. -->

### Detailed Explanation of Attention for NMT

We have an input sequence $(x_1, x_2, x_3, ..., x_n)$ and let their corresponding encoder hidden states be $(h_1, h_2, h_3, ..., h_n)$. Let $s_i$ be the _decoder_ hidden state at $i^{th}$ decoding step. Then the context vector $c_i$ that will be used to predict the word $y_i$ can be obtained as follows:
1. $e_{ij} = f(s_i, h_j)$, here $h_j$ is the encoder hidden state of the $j^{th}$ sequence input, $s_i$ is the decoder hidden state at the $i^{th}$ decoding step.
2. We get $e_{11}, e_{12}, e_{13}, ..., e_{1n}$ from  $f(s_1, h_1), f(s_1, h_2), f(s_1, h_3), ..., f(s_1, h_n)$ respectively.
3. These $e_{ij}$ are unnormalized scores/weights. Apply softmax to get normalized scores/weights: $a_{ij} = \frac{e_{ij}}{\sum_{j=1}^{T_{n}}e_{ij}}$.
4. To calculate the context vector $c_i$ for the $i^{th}$ decoding step, take average of encoder hidden states with $a_{ij}$ as weights: $c_i = \sum_{j=1}^{T_n} a_{ij}h_j$.

Now with context vector $c_i$, decoder hidden state $s_i$ and last predicted word $y_{i-1}$, the decoder will predict the next word $y_i$ as per following expression: 

\begin{equation}
y_i = g(y_{i-1}, s_i, c_i)
\end{equation}

This is the alignment/attention model as described in the paper.


<!--
Let's take an example of Machine Translation from English to Spanish. Let's say we have one input sentence in `en` and the corresponding output sentence in `es`. If we are trying to translate the given input sentence, $e_{ij}$, and in turn $a_{ij}$, will tell us how much the $j_{th}$ input word is important for predicting the $i^{th}$ word in output sentence. The alignment model will learn to calculate the weights or importance from $(i-1)^{th}$ output word and $j^{th}$ input word. -->

<!--
## Questions

  1. You are given two MT tasks: `en -> de` and `fr -> es`. Without training a model, is it possible to know, which language pair model will be able to handle larger input sequences?
  
  2. If I train any rnn on large sequences as well, will it be able to handle inputs in the similar length range?
  
  3. Can I use a single attention layer after each feature extraction layer?
    
    1. A, BiLSTM ==> [BiLSTM1, A, BiLSTM2, A, BiLSTM3, A, BiLSTM4, A] # W = lower
    
    2. A, BiLSTM ==> [BiLSTM1, A1, BiLSTM2, A2, BiLSTM3, A3, BiLSTM4, A4]  # W = higher
-->


This is a developed in collaboration with [@mandalbiswadip](https://github.com/mandalbiswadip) 


