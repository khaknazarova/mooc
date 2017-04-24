import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_class = W.shape[1]
  dscores = np.zeros((num_train, num_class))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        score = X[i].dot(W)
        sum_of_exp = 0
        for j in range(num_class):
            sum_of_exp += np.exp(score[j])
            dscores[i][j] = np.exp(score[j])
        loss += -np.log(np.exp(score[y[i]])/sum_of_exp)
        dscores[i] = dscores[i]/sum_of_exp
        dscores[i][y[i]] -=1
        
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  loss /= num_train
  
  loss += 0.5*reg*np.sum(W * W)
  dW +=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_class = W.shape[1]
  dscores = np.zeros((num_train, num_class))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = np.exp(X.dot(W))
  correct_class_score = [score[i,y[i]] for i in range(num_train)]
  sum_of_one_scores = np.sum(score, axis = 1)
  loss -= np.sum(np.log(correct_class_score/sum_of_one_scores))
  loss /= num_train
  loss += 0.5*reg*np.sum(W * W)
  
  Y = np.zeros((num_train, num_class))
  for i in range(num_train):
      Y[i][y[i]] = 1
  dscores = np.array([score[i]/sum_of_one_scores[i] for i in range(num_train)]) - Y
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  dW +=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

