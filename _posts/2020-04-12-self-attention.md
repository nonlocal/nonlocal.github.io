# A simple Explanation of Self Attention 

1. TOC
{:toc}

## Prerequisite

Please read this post before moving ahead: [attention in simple terms](https://nonlocal.github.io/2020/04/07/attention.html)


## Background

In [this blog](https://nonlocal.github.io/2020/04/07/attention.html), we reviewed attention in the context of encoder-decoder network where given the current decoder hidden state (**query**), the attention model "probes" over all of the encoder hidden states (**values**), to calculate attention/context vector for the current decoding step.

In this blog, we are going to review another type of attention mechanism, where attention vector is obtained from the query and the values of the same layer, known as self-attention. For a given hidden state, the attention vector in this case "attends to" all the hidden states from the same layer.

Let me illustrate. 

## Graphical Representation

![](/images/Attention.jpg)
*Figure 1: Self-attention mechanism to calculate context vector for the first sequence element.*
{: style="text-align: center;"}


## Explanation
Let's take a closer look at the image above and try to understand what is going on.
 
1. We have a sequence of inputs $X = (x_1, x_2, x_3, ..., x_n)$.
2. Specifically here, we are using an RNN to model the sequence $X$. This argument/approach is general though. (Will take another example below without the RNN as sequence model).
3. The initial hidden state of the RNN is $h_0$ and for each element of $X$, the RNN outputs a sequence of hidden states $H = (h_1, h_2, h_3, ..., h_n)$.
4. For each hidden state given by the RNN, the attention model calculates an attention/context vector, $C = (c_1, c_2, c_3, ..., c_n)$ and concatenates it with its corresponding hidden state. The concatenated (hidden state and context vector) serves as the input to the next layer: $X^{next} = conc(H, C) = ([h_1, c_1], [h_2, c_2], [h_3, c_3], ..., [h_n, c_n])$.

    Let's calculate the attention vector $c_1$. The same process can be applied to calculate the rest of the attention/context vectors. The process followed is as described in [this blog](https://nonlocal.github.io/2020/04/07/attention.html). Please make sure you have understood it. It's THE prerequisite for this blog.

    a. The hidden state $h_1$ is shared across all timesteps of the sequence as can be seen in the <span style="color:blue">blue colored arrows</span>.
    b. 
