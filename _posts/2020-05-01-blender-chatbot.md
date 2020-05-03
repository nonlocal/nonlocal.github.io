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




Questions:
1. Minimum length constrain implementation : sub-sampling the vocab without the <END> token until we have sampled a sequence of length L-1??
2. 
