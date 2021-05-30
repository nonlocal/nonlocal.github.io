# Adverserial Examples

1. TOC
{:toc}

## Definition

What are adverserial examples in Deep Learning?


Consider a deep neural network (DNN) that generalizes well on some task. It can be object recognition or intent classification. We expect such network to be robust to small perturbations of its input, because small perturbation cannot change the output of the deep neural network. But it has been observed that even small perturbations to the input causes the model to misclassify. 

The perturbations that make the model misclassify an input are often non-random and can be found via different optimization strategies see for examples [this](https://arxiv.org/pdf/1412.6572.pdf) and [this](https://arxiv.org/pdf/1607.02533.pdf). The inputs obtained after such perturbations are termed [“adversarial examples”](https://arxiv.org/pdf/1312.6199.pdf).


## Are adverserial examples unique for a given dataset and a model?

For a task $T$, let's say we train a deep neural network $f1$ on a subset $D1$  of training data $D$ with training procedure $P1$ (the training procedure consists of optimization of the network with backpropagation(SGD/Adam/etc), batch size during training, number of epochs, etc). Let's say we use this network to generate a set of adverserial examples $R1$. 

For the same task $T$ , let's train a different deep neural network $f2$ (different hyperparameters, layers sizes, activations or even the entire architecure) using the training procedure $P2$ (with the same or different optimizer, same of different batch size/ number of epochs) on a different subset $D2$ of training data $D$. 

Will the perturted samples generated above (R1) be adverserial to this new model $f2$? The answer is an emphatic _Yes!_. [Emperically](https://arxiv.org/pdf/1312.6199.pdf), it has been found that the set of adverserial examples $R1$ generated above will also be adverserial to the new model f2 trained for the same task T on a different subset of the data D2 or with different training procedure. 

## Are adverserial examples universal?

We now know that adverserial examples are unique, given a task. But how is this possible?

Consider a linear classifier $h(x) = w^T.x$, where the prediction will be $sign(h)$. For example,  Ham if $+1$ or spam if $-1$.

If we add a small perturbation $\eta$  to input $x$ to create a new input $\tilde{x} = x + \eta$ , we expect the classifier to assign the same label to this new input as long as the small perturbation $\eta$ is smaller than the precision(smallest possible change that can be detected in the data) of the input. That is, we expect

$h(x) = h(\tilde{x} = x + \eta) = w^T.\tilde{x} = w^T.x + w^T.\eta$

as long as $max(\eta)<\epsilon$ where $\epsilon$ is the precision of the input.

Here we see that the value of the perturbation causes the function $h$ to grow by $w^T.\eta$. We can put an upper limit on this change by selecting $\eta = sign(w)* \epsilon$. Assuming that the average magnitude of the weight vector $w$ is m, the growth in the function $h$ can be approximated by $\epsilon mn$ where $n$ is the dimensionality of the input.

Notice that the change depends on $n$, the dimentionality of the input. So, for tasks where the feature dimension is very large, even an infinitesimal perturbation can cause a very large change in the value of the output funtion $h$. This can result in different label prediction for very similar inputs $x$ and $\tilde{x}$.

From this analysis we see that, if the feature dimension is very high, then we can have adverserial examples for _any type_ of classifier. 


This is not specific to _just_ deep neural networks. But deep neural networks are more prone to adverserial examples because the activation functions used in deep neural networks cause the representation manifold to become "step-wise" (since the activate functions are themselves step-wise) and non-smooth (whenever there's a step in the activation function).




<!-- p2pdistance + p2hpdistance -->

## Conclusion
1. Curse of dimensionality strikes again! Adverserial examples are properties of systems with higher dimensions. 
2. Adverserial examples seen in deep neural networks are mainly due to the activation functions used.

This post was developed in collaboration with [Biswadip Mandal](https://github.com/mandalbiswadip).


