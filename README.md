# Implementation of neural network layers and optimizers in Python 📝

## Structure 
Implementations of a few neural network layers: 
- [Linear](./src/layers/linear.py) - Linear (fully connected) layer 
- [ReLU](./src/layers/relu.py) - Layer ReLU activation function 
- [Sigmoid](./src/layers/sigmoid.py) - Sigmoid activation function layer 
- [Dropout](./src/layers/dropout.py) - Dropout layer (random dropout of neurons) 
- [LogSoftmax] (./src/layers/log_softmax.py) - LogSoftmax layer (for multiclass classification) 
- [RNN](./src/layers/rnn.py) - Recurrent layer (Vanilla RNN) 
- [GRU](./src/ layers/gru.py) - Recurrent GRU layer 

Available loss functions:
- [NLLLoss](./src/criterions/neg_log_likelihood_loss.py) - Negative Log Likelihood Loss 
- [CrossEntropyLoss](./src/criterions/cross_entropy_loss.py) - CrossEntropyLoss represented as LogSoftmax + NLLLoss 
- [FocalLoss](./ src/criterions/focal_loss.py) - FocalLoss: $-(1-p)^\gamma * log(p)$ 

Optimizers: 
- [SGD](./src/optimizers/sgd.py) - Stochastic Gradient Descent method 
- [AdaGrad](./src/optimizers/adagrad.py) - AdaGrad method 
- [RMSprop](./src/optimizers/rmsprop.py) - RMSprop method