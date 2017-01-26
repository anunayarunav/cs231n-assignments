import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.fc_net import *
class Convnet(object):

    """
    A complete convnet package which follows as
    conv-relu-pool x N - affine-batchnorm-relu-dropout x M - affine - softmax/svm
    """
    def __init__(self, conv_filters=[16, 16, 16, 16], pool = [0, 1, 0, 1], filter_size = 7, hidden_dims = [100], 
                 input_dim=(3, 32, 32), num_classes=10, dropout=0, use_batchnorm=False, 
                 reg=0.0, use_svm=False, weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialization of a convnet.

        Inputs.
        - conv_filters: A list of filters numbers for each conv net layer.
        - pool: whether to use pooling after ith convnet or not (same size as conv_filters)
        - filter_size: filter sizes for each filter
        - hidden_dims: number of hidden dimensions after convolutions
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        self.hidden_dims = hidden_dims
        self.conv_filters = conv_filters

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in convW1 and convb1; for the second layer use W2 and b2, etc. Weights   #
        # should be initialized from a normal distribution with standard deviation #
        # equal to weight_scale and biases should be initialized to zero.          #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        C, H, W = input_dim
        HH = filter_size
        WW = filter_size
        self.use_pooling = {}
        dim = H*W
        F = C
        for i in xrange(len(self.conv_filters)):
            j = i+1
            F = conv_filters[i]
            self.params['convW%d'%j] = weight_scale*np.random.randn(F, C, HH, WW)
            self.params['convb%d'%j] = np.zeros(F)
            C = F
            self.use_pooling[i] = pool is None or pool[i]
            if self.use_pooling[i]:
                dim = dim/4
        
        dim = dim*F
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
          self.params[k] = v.astype(dtype)
        
        #create second network for affine processing
        self.fc_net = FullyConnectedNet(hidden_dims, input_dim=dim, num_classes=num_classes,
                                        dropout=dropout, use_batchnorm=use_batchnorm, reg=reg,
                                        weight_scale=weight_scale, use_svm=use_svm,dtype=dtype, plugin=True, seed=seed)
        
        # transfer all params of fc net
        for k, v in self.fc_net.params.iteritems():
            self.params[k] = v
            
    
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        
        #Transfer updated params of this model to fc_net
        for k, v in self.fc_net.params.iteritems():
            self.fc_net.params[k] = self.params[k]
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the convnet, computing              #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # pass pool_param to the forward pass for the max-pooling layer
        cache = {}
        out = X
        for i in xrange(len(self.conv_filters)):
            j = i+1
            w, b = self.params['convW%d'%j], self.params['convb%d'%j]
            filter_size = w.shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            if self.use_pooling[i]:
                pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
                out, cache[j] = conv_relu_pool_forward(out, w, b, conv_param, pool_param)
            else:
                out, cache[j] = conv_relu_forward(out, w, b, conv_param)
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # If test mode return early

        if y is None:
          scores = self.fc_net.loss(out)
          return scores
        
        reg = self.reg
        loss, grads, dx = self.fc_net.loss(out, y)
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        for i in xrange(len(self.conv_filters), 0, -1):
            if self.use_pooling[i-1]:
                dx, dw, db = conv_relu_pool_backward(dx, cache[i])
            else:
                dx, dw, db = conv_relu_backward(dx, cache[i])
                
            #regularize
            w = self.params['convW%d'%(i)]
            dw += reg * w
            loss += 0.5 * reg * np.sum(w*w)
            
            grads['convW%d'%(i)] = dw
            grads['convb%d'%(i)] = db
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads