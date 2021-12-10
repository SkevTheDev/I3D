#I3D architecture
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class InceptionI3d(object):

  """Inception-v1 I3D architecture."""
  # Endpoints of the model in order. During construction, all the endpoints up
  # to a designated `final_endpoint` are returned in a dictionary as the
  # second return value.

  VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions',
  )

  def __init__(self, num_classes=2, spatial_squeeze=True,
               final_endpoint='Predictions', name='inception_i3d'): #final_endpoint='Logits'

    """Initializes I3D model instance.

    Args:
      num_classes: The number of outputs in the logit layer (default 400, which matches the Kinetics dataset).
      spatial_squeeze: Whether to squeeze the spatial dimensions for the logits before returning (default True).
      final_endpoint: The model contains many possible endpoints.
          `final_endpoint` specifies the last endpoint for the model to be built up to. In addition to the output at `final_endpoint`, all the outputs
          at endpoints up to `final_endpoint` will also be returned, in a dictionary. `final_endpoint` must be one of
          InceptionI3d.VALID_ENDPOINTS (default 'Logits').
      name: A string (optional). The name of this module.

    Raises:
      ValueError: if `final_endpoint` is not recognized.
    """
       
    print("\nn _init_, InceptionI3D.py")

    if final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % final_endpoint)

    super(InceptionI3d, self).__init__() #name=name)
    self._num_classes = num_classes
    self._spatial_squeeze = spatial_squeeze
    self._final_endpoint = final_endpoint
       
  def _build(self, is_training, dropout_keep_prob=1.0):

    """Connects the model to inputs.
       
    Args:
      inputs: Inputs to the model, which should have dimensions `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
      is_training: whether to use training mode for snt.BatchNorm (boolean).
      dropout_keep_prob: Probability for the tf.nn.dropout layer (float in [0, 1)).
    Returns:
      A tuple consisting of:
        1. Network output at location `self._final_endpoint`.
        2. Dictionary containing all endpoints up to `self._final_endpoint`, indexed by endpoint name.
    Raises:
      ValueError: if `self._final_endpoint` is not recognized.
    """
    print("\nin _build(), InceptionI3D.py")

    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError('Unknown final endpoint %s' % self._final_endpoint)
  
    end_points = {}
    
    inputs = tf.keras.layers.Input((20, 64, 64, 3)) 
    print("inputs.shape = " + str(inputs.shape)) #(None, 20, 64, 64, 3)
    net = inputs

    end_point = 'Conv3d_1a_7x7'
    #net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7], stride=[2, 2, 2], name=end_point)(net, is_training=is_training)    
    net = tf.keras.layers.Conv3D(64, (7, 7, 7), (2, 2, 2), use_bias=True)(net) 
    net = tf.keras.layers.BatchNormalization(axis=-1)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_2a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Conv3d_2b_1x1'
    #net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name=end_point)(net, is_training=is_training)
    net = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
    net = tf.keras.layers.BatchNormalization(axis=-1)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)    
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Conv3d_2c_3x3'
    #net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3], name=end_point)(net, is_training=is_training)
    net = tf.keras.layers.Conv3D(192, (3, 3, 3), (1, 1, 1), use_bias=True)(net) 
    net = tf.keras.layers.BatchNormalization(axis=-1)(net)
    net = tf.keras.layers.Activation(tf.nn.relu)(net)    
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_3a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME', name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_3b'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(96, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)
    
        #branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)        
        branch_1 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    
        
      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(16, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)
    
        #branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)      
        branch_2 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    
        
      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    
      
      print("\nbranch0.shape = ", branch_0.shape)
      print("branch1.shape = ", branch_1.shape)
      print("branch2.shape = ", branch_2.shape)
      print("branch3.shape = ", branch_3.shape)      

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
   
    end_point = 'Mixed_3c'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)
    
        #branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(192, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)
    
        #branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(96, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)
    
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_4a_3x3'
    net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4b'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(192, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(96, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    
        
        #branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(208, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(16, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(48, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    
      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points
    
    end_point = 'Mixed_4c'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(160, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(112, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(224, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(24, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4d'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(256, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(24, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(24, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4e'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(112, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(144, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(288, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(64, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_4f'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(256, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(160, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(320, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(24, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'MaxPool3d_5a_2x2'
    net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name=end_point)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5b'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(256, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(160, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(320, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(32, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3], name='Conv3d_0a_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    end_point = 'Mixed_5c'
    with tf.compat.v1.variable_scope(end_point):
      with tf.compat.v1.variable_scope('Branch_0'):
        #branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_0 = tf.keras.layers.Conv3D(384, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_0 = tf.keras.layers.BatchNormalization(axis=-1)(branch_0)
        branch_0 = tf.keras.layers.Activation(tf.nn.relu)(branch_0)    

      with tf.compat.v1.variable_scope('Branch_1'):
        #branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(192, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)    

        #branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_1, is_training=is_training)
        branch_1 = tf.keras.layers.Conv3D(384, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_1) #(3, 3, 3)
        branch_1 = tf.keras.layers.BatchNormalization(axis=-1)(branch_1)
        branch_1 = tf.keras.layers.Activation(tf.nn.relu)(branch_1)

      with tf.compat.v1.variable_scope('Branch_2'):
        #branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1], name='Conv3d_0a_1x1')(net, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(48, (1, 1, 1), (1, 1, 1), use_bias=True)(net) 
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)    

        #branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3], name='Conv3d_0b_3x3')(branch_2, is_training=is_training)
        branch_2 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_2) #(3, 3, 3)
        branch_2 = tf.keras.layers.BatchNormalization(axis=-1)(branch_2)
        branch_2 = tf.keras.layers.Activation(tf.nn.relu)(branch_2)

      with tf.compat.v1.variable_scope('Branch_3'):
        branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME', name='MaxPool3d_0a_3x3')
        #branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1], name='Conv3d_0b_1x1')(branch_3, is_training=is_training)
        branch_3 = tf.keras.layers.Conv3D(128, (1, 1, 1), (1, 1, 1), use_bias=True)(branch_3) 
        branch_3 = tf.keras.layers.BatchNormalization(axis=-1)(branch_3)
        branch_3 = tf.keras.layers.Activation(tf.nn.relu)(branch_3)    

      net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    end_points[end_point] = net
    if self._final_endpoint == end_point: return net, end_points

    print("\nbefore average pool, net.shape = ", net.shape); # (None, 2, 2, 2, 1024)
    end_point = 'Logits'
    with tf.compat.v1.variable_scope(end_point):
      net = tf.nn.avg_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='VALID') #ksize=[1, 2, 7, 7, 1]
      print("after average pool, net.shape = ", net.shape); # (None, 1, 1, 1, 1024)
      net = tf.nn.dropout(net, 0.2) #dropout_keep_prob
      print(net.shape) #(None, 1, 1, 1, 1024)
      """
      logits = Unit3D(output_channels=self._num_classes,
                      kernel_shape=[1, 1, 1],
                      activation_fn=None,
                      use_batch_norm=False,
                      use_bias=True,
                      name='Conv3d_0c_1x1')(net, is_training=is_training)
      """
      logits = tf.keras.layers.Conv3D(self._num_classes, (1, 1, 1), (1, 1, 1), use_bias=True)(net) #(3, 3, 3)
      print("logits.shape = ", logits.shape);# (None, 1, 1, 1, 2)

      if self._spatial_squeeze:
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
      print("after squeeze, logits.shape = ", logits.shape); # (None, 1, 2)
          
    averaged_logits = tf.reduce_mean(logits, axis=1)
    print("tf.reduce_mean, averaged_logits.shape = ", averaged_logits.shape); # (None, 2)
    end_points[end_point] = averaged_logits
    #print("\nend_points = ", end_points)
    if self._final_endpoint == end_point: return averaged_logits, end_points

    end_point = 'Predictions'
    predictions = tf.nn.softmax(averaged_logits)
    print("after softmax, predictions.shape = ", predictions.shape); # (None, 2)
    end_points[end_point] = predictions
    print("\nend_points = ", end_points)
    #return predictions, end_points
    
    model = tf.keras.Model(inputs, predictions, name='i3d')     
    return model