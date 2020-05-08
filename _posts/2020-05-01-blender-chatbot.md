# The Blender Chatbot


Please find the paper here : [Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf)

I am extremely curious about reviewing this paper. I have had a lot of bruises building chatbots for my [current company](https://www.gopaysense.com/). I have noticed a few thing about building chatbots in _The Real World&trade;_. 

My primary goal with reading this paper is to find out if it addresses _The Real World Problems&trade;_. 

The secondary goal is to understand the training data, the modelling algorithm(s), and situations where these settings are helpful and where they are not.

For example: 
1. Can this approach work for low resource languages? 
1. Can I use the same approach to train an open domain Hindi chatbot? 
1. Can this approach be used to transfer the learning from an open domain English chatbot to an open domain French/Hindi/Spanish chatbot?
1. What about a multi-lingual chatbot? 


One counter-intuitive application would be the usage of this chatbot for a closed domain : train the bot in the "open domain" setting but use it for a very specific purpose eg booking an airline ticket, checking up on the health of a patient, banking, paying bills etc just to name a few.

1. TOC
{:toc}



## Model Architecture
1. Retrieval
2. Generative
3. Retrieve And Refine

### Retrieval

Dialogue history as input. The model predicts the next utterance by scoring a large set of candidate responses. 


### Generator

Seq2Seq Transformer is used to generate responses as opposed to choosing from a fixed set.

### Retrieve And Refine

Generative models often produce dull and repetitive responses. The solution is to add a retrieval step before generation.

#### Dialogue Retrieval
The Retrieval model above is used to retrieve a response. This response along with the dialogue history is fed to the generator model above to generate a response.

#### Knowledge Retrieval
Instead of retriving a response from a set of reponses, we retrieve from a large knowledge base. Then the dialogue history along with this retrived knowledge is fed to the generator model to generate a response. 


## Training

### Retrieval

Some special cross entropy loss

### Generator

Maximum Likelihood Estimation


### Retrieve And Refine

For this model, simply appending the retrieval model responses to generator model input and training it with MLE does not work. (Why?) Because R&R model learns to ignore the output of the retrieval model. Instead there is some blending between retrieval response and gold response. (This blending is used only for Dialogue based R&R model.)

### Unlikelihood Training

To address the failures of training models, include unlikelihood loss as well.

In this case the total loss is likelihood loss + unlikelihood loss.

Likelihood loss optimizes for the given probabilty distribution and unlikelihood loss corrects the biases.


## Decoding

### Beam Search


### Sampling
Restrict sampling of vocabulary to a subset using a model.
1. Top K Sampling
2. Sample and Rank

### Response length
#### Minimum length
Do not generate the <END> token until the minimum sequence length is achieved.
  
#### Predictive length
Predict the length of the response. Train on human-to-human conversational data to predict the length of the next response : < 10, < 20,< 30 and > 30 tokens. The architecture for this 4 way classification is same as retrieval model.   

#### Subsequence blocking

Questions:
1. Minimum length constrain implementation : sub-sampling the vocab without the <END> token until we have sampled a sequence of length L-1??
2. 


