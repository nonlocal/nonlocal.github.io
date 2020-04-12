# Self Attention

In [this blog](https://nonlocal.github.io/2020/04/07/attention.html), we reviewed attention in the context of encoder-decoder network where given the current decoder hidden state (**query**), the attention model "probes" over all of the encoder hidden states (**values**), to calculate attention/context vector for the current decoding step.

In this blog, we are going to review another type of attention mechanism, where attention vector is obtained from the query and the values of the same layer, known as self-attention. For a given hidden state, the attention vector in this case "attends to" all the hidden states from the same layer.

Let me illustrate. 

![](/images/Attention.jpg "Self Attention")
*Figure 1: Self-attention.*
{: style="text-align: center;"}
