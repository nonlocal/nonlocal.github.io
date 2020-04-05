# Voice and Accompaniment Separation in Music

1. TOC
{:toc}

This is a review of the paper [Voice and accompaniment separation in music using self-attention convolutional neural network](https://arxiv.org/pdf/2003.08954v1.pdf)

Reader is assumed to be familiar with basic constructs like CNN, LSTM, Attention, etc.

## Introduction

### Automatic Karoake / Remixing
Let's say there is a new song released by one of your favourite musician/artist. 
1. You want to ~~cry~~ sing your heart out to it but you only need the music from that song, you want to sing the lyrics yourself. What do you do? 
2. Let's say you are an up-and-coming DJ. You want to keep the original lyrics of the singer in the song as is, but you want to add/mix your own music to show off to your audience.


## Problem

How do you do it? How do you separate music from lyrics or vice-versa?

In general, how do you separate _one_ source of voice from _another_? 


## Solution

1. Base network : Dense-UNet
2. Improvement : Self-attention subnets on base network 

### Dense-UNet voice and accompaniment separation
In CV, UNet is used for semantic segmentation.

Let $X_1(t, f)$ denote Short-Time Discrete Fourier Transform (STFT) of human voice where $t$ and $f$ are time and frequency indices. Let X_2(t, f) denote STFT of accompaniment. Then the music mixture can be described as :

\begin{equation}
Y(t, f) = X_1(t, f) + X_2(t, f)\tag{abc}
\end{equation}




## References




## Neural Network Architectures mentioned in the paper

1. UNet-CNN
2. Skip-UNet-CNN
3. Dense-UNet-CNN
4. MMDenseLSTM
