import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C = input_dim[0]
    H = input_dim[1]
    W = input_dim[2]
    F = num_filters
    stride_conv = 1  # stride
    P = (filter_size - 1) / 2  # padd
    HH = (H + 2 * P - filter_size) / stride_conv + 1
    WW = (W + 2 * P - filter_size) / stride_conv + 1
    
    W1 = np.random.normal(0, weight_scale, size=F*C*filter_size*filter_size).reshape(F, C, filter_size, filter_size)
    b1 = np.zeros(F)
    
    H_pool = (HH - 2) / 2 + 1
    W_pool = (WW - 2) / 2 + 1
    
    W2 = np.random.normal(0, weight_scale, size=F*H_pool*W_pool*hidden_dim).reshape(F*H_pool*W_pool, hidden_dim)
    b2 = np.zeros(hidden_dim)
    
    W3 = np.random.normal(0, weight_scale, size=hidden_dim*num_classes).reshape(hidden_dim, num_classes)
    b3 = np.zeros(num_classes)
    
    self.params['W1']=W1
    self.params['b1']=b1
    self.params['W2']=W2
    self.params['b2']=b2
    self.params['W3']=W3
    self.params['b3']=b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_conv, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
    out_relu1, cache_relu1 = relu_forward(out_conv)
    out_pool, cache_pool = max_pool_forward_naive(out_relu1, pool_param)
    out_affine1, cache_affine1 = affine_relu_forward(out_pool, W2, b2)
    out_affine2, cache_affine2 = affine_forward(out_affine1, W3, b3)
    
    scores = out_affine2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = data_loss + reg_loss
    
    affine_dx3, affine_dW3, affine_db3 = affine_backward(dscores, cache_affine2)
    
    grads['W3'] = affine_dW3 + self.reg * self.params['W3']
    grads['b3'] = affine_db3
    
    affine_dx2, affine_dW2, affine_db2 = affine_relu_backward(affine_dx3, cache_affine1)
    
    grads['W2'] = affine_dW2 + self.reg * self.params['W2']
    grads['b2'] = affine_db2
    
    pool_dx1 = max_pool_backward_naive(affine_dx2, cache_pool)
    
    conv_dx = relu_backward(pool_dx1, cache_relu1)
    
    dx, dW1, db1 = conv_backward_naive(conv_dx, cache_conv)
    
    grads['W1'] = dW1 + self.reg * self.params['W1']
    grads['b1'] = db1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
