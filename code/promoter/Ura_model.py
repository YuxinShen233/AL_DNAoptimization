import warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow.keras as keras  ## important to make sure non tf.keras is hidden

### Reference to helpful open sourced libraries utilized in this project : 
##https://github.com/CyberZHG ### Thanks CyberZHG ! 
#http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=9 
#from keras_multi_head import MultiHeadAttention , MultiHead
#from keras_position_wise_feed_forward import FeedForward
#from keras_layer_normalization import LayerNormalization 


import argparse,pwd,os,numpy as np,h5py
from os import makedirs
from os.path import splitext,exists,dirname,join,basename , realpath
import multiprocessing as mp, ctypes
import time , csv ,pickle  , matplotlib  , multiprocessing,itertools
import seaborn as sns
import os, gc , datetime , sklearn , scipy , pydot , random  
from tqdm import tqdm 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.client import device_lib
from tensorflow.keras import Input
from tensorflow.keras.layers import  Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten , Conv1D, Concatenate , Permute
from tensorflow.keras.layers import Bidirectional,LSTM,CuDNNLSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Add , LeakyReLU ,Reshape , Activation , MaxPooling1D , Lambda , Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.backend import conv1d
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import h5py , tensorflow , re
import tensorflow as tf, sys, numpy as np, h5py, pandas as pd
from tensorflow import nn
from tensorflow.contrib import rnn
from os import makedirs
from tensorflow.keras.utils import multi_gpu_model
import glob , math
import time , base64 , copy



if "platform" in os.environ:
    if os.environ['platform'] == 'streamlit_sharing' : 
        path_prefix = './app/'
else :
    path_prefix = ''


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def fitness_function_model(model_params) :

    n_val_epoch = model_params['n_val_epoch']
    epochs= model_params['epochs']
    batch_size= model_params['batch_size']
    l1_weight= model_params['l1_weight']
    l2_weight= model_params['l2_weight']
    motif_conv_hidden= model_params['motif_conv_hidden']
    conv_hidden= model_params['conv_hidden']
    n_hidden= model_params['n_hidden']
    n_heads= model_params['n_heads']
    conv_width_motif= model_params['conv_width_motif']
    dropout_rate= model_params['dropout_rate']
    attention_dropout_rate= model_params['attention_dropout_rate']
    lr= model_params['lr']
    n_aux_layers= model_params['n_aux_layers']
    n_attention_layers= model_params['n_attention_layers']
    add_cooperativity_layer= model_params['add_cooperativity_layer']
    device_type = model_params['device_type']
    input_shape = model_params['input_shape']
    loss = model_params['loss']


    
    if(model_params['device_type']=='tpu'):
        input_layer = Input(batch_shape=(batch_size,input_shape[1],input_shape[2]))  #trX.shape[1:] #batch_shape=(batch_size,110,4)

    else :
        input_layer = Input(shape=input_shape[1:])  #trX.shape[1:] #


    #https://arxiv.org/pdf/1801.05134.pdf

    x_f,x_rc = rc_Conv1D(motif_conv_hidden, conv_width_motif, padding='same' , \
               kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
              data_format = 'channels_last' , use_bias=False)(input_layer)
    x_f = BatchNormalization()(x_f)
    x_rc = BatchNormalization()(x_rc)

    x_f = Activation('relu')(x_f)
    x_rc = Activation('relu')(x_rc)


    if(add_cooperativity_layer==True) : 
        x_f = Lambda(lambda x : K.expand_dims(x,axis=1))(x_f)
        x_rc = Lambda(lambda x : K.expand_dims(x,axis=1))(x_rc)

        x =Concatenate(axis=1)([x_f, x_rc] )

        x = keras.layers.ZeroPadding2D(padding = ((0,0 ),(int(conv_width_motif/2)-1,int(conv_width_motif/2))), 
                                          data_format = 'channels_last')(x)
        x = Conv2D(conv_hidden, (2,conv_width_motif), padding='valid' ,\
               kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
              data_format = 'channels_last' , use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda x : K.squeeze(x,axis=1))(x)
        


    else:
        x =Add()([x_f, x_rc] )
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)


    for i in range(n_aux_layers) : 
        #res_input = x
        x = Conv1D(conv_hidden, (conv_width_motif), padding='same' ,\
               kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
              data_format = 'channels_last' , use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #x = Add()([res_input, x])


        
    for i in range(n_attention_layers) : 
        mha_input = x
        x = MultiHeadAttention( head_num=n_heads,name='Multi-Head'+str(i),
                              kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight))(x) #### DO NOT MAX POOL or AVG POOL 
        if dropout_rate > 0.0:
            x = Dropout(rate=attention_dropout_rate)(x)
        else:
            x = x
        x = Add()([mha_input, x])
        x = LayerNormalization()(x)
        
        ff_input = x
        x  = FeedForward(units= n_heads, kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight))(x)
        if dropout_rate > 0.0:
            x = Dropout(rate=attention_dropout_rate)(x)
        else:
            x = x
        x = Add()([ff_input, x])
        x = LayerNormalization()(x)    



    x = Bidirectional(LSTM(n_heads, return_sequences=True,
                           kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight),
                           kernel_initializer='he_normal' , dropout = dropout_rate))(x)
    x = Dropout(dropout_rate)(x)


    if(len(x.get_shape())>2):
        x = Flatten()(x) 

    x = Dense(int(n_hidden), 
                    kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                    kernel_initializer='he_normal' , use_bias=True)(x)
    x = Activation('relu')(x) 
    x = Dropout(dropout_rate)(x) #https://arxiv.org/pdf/1801.05134.pdf


    x = Dense(int(n_hidden), kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                    kernel_initializer='he_normal', use_bias=True )(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x) #https://arxiv.org/pdf/1801.05134.pdf

    output_layer = Dense(1, kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                    activation='linear', kernel_initializer='he_normal', use_bias=True )(x) 


    model = Model(input_layer, output_layer)
    opt = tf.train.RMSPropOptimizer(lr) #tf.keras.optimizers.Adam(lr=lr)#
    

    model.compile(optimizer=opt, loss=loss,metrics=[r_square]) 
    
    return model

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.
        :param units: Dimension of hidden units.
        :param activation: Activation for the first linear transformation.
        :param use_bias: Whether to use the bias term.
        :param kernel_initializer: Initializer for kernels.
        :param bias_initializer: Initializer for kernels.
        :param kernel_regularizer: Regularizer for kernels.
        :param bias_regularizer: Regularizer for kernels.
        :param kernel_constraint: Constraint for kernels.
        :param bias_constraint: Constraint for kernels.
        :param kwargs:
        """
        self.supports_masking = True
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y
 
class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
 
class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.
    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(key_len), axis=0)
            upper = K.expand_dims(K.arange(query_len), axis=-1)
            e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)
        if self.return_attention:
            return [v, a]
        return v
    
 
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """

        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only
        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': int(self.head_num),
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y
    



class rc_Conv1D(Conv1D):

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[1],
                                               self.kernel_size[0],
                                               padding=self.padding,
                                               stride=self.strides[0])
        return [(int(input_shape[0]), int(length), int(self.filters)),
                (int(input_shape[0]), int(length), int(self.filters))]

    def call(self, inputs):
        #create a rev-comped kernel.
        #kernel shape is (width, input_channels, filters)
        #Rev comp is along both the length (dim 0) and input channel (dim 1)
        #axes; that is the reason for ::-1, ::-1 in the first and second dims.
        #The rev-comp of channel at index i should be at index i
        revcomp_kernel =\
            K.concatenate([self.kernel,
                           self.kernel[::-1,::-1,:]],axis=-1)
        if (self.use_bias):
            revcomp_bias = K.concatenate([self.bias,
                                          self.bias], axis=-1)

        outputs = K.conv1d(inputs, revcomp_kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs += K.bias_add(outputs,
                                  revcomp_bias,
                                  data_format=self.data_format)

        if (self.activation is not None):
            outputs = self.activation(outputs)
        x_f = outputs[:,:,:int(outputs.get_shape().as_list()[-1]/2)]
        x_rc = outputs[:,:,int(outputs.get_shape().as_list()[-1]/2):]

        return [x_f,x_rc]
 
def load_model(model_conditions ) : 
    NUM_GPU = len(get_available_gpus())
    dir_path=os.path.join(path_prefix+'models',model_conditions)
    model_path=os.path.join(dir_path,"fitness_function.h5")

    ### Load the parameters used for training the model
    f = open(os.path.join(dir_path,'model_params.pkl'),"rb")
    model_params = pickle.load(f)
    batch_size = model_params['batch_size']
    f.close()


    
    ### Load the model on multiple CPU/GPU(s) with the largest possible batch size
    scaler= sklearn.externals.joblib.load(os.path.join(dir_path,'scaler.save'))
    model_params['batch_size'] = np.power(2,10 + NUM_GPU)
    batch_size = model_params['batch_size']
    model_params['device_type'] = 'gpu'
    model = fitness_function_model(model_params)
    model.load_weights(model_path)
    if NUM_GPU > 1 :
        model = tf.keras.utils.multi_gpu_model(model,NUM_GPU,cpu_merge=True,cpu_relocation=False)

    if 0 : #Change to 1 if using TPU ## Changing the batch size on using the tf.keras.models.load_model is not permitted,but TPU needs this
        scaler= sklearn.externals.joblib.load(os.path.join(dir_path,'scaler.save'))
        batch_size = model_params['batch_size']
        model_params['device_type'] = 'tpu'
        model = fitness_function_model(model_params)
        model.load_weights(model_path)
        
        if(model_params['device_type']=='tpu'):
            tpu_name = os.environ['TPU_NAME']
            tpu_grpc_url = TPUClusterResolver(tpu=[tpu_name] , zone='us-central1-a').get_master()
            if(tpu_grpc_url) : 
                model = tf.contrib.tpu.keras_to_tpu_model(model,
                        strategy=tf.contrib.tpu.TPUDistributionStrategy(
                            tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)))

            if 0 : 
                model = tensorflow.keras.models.load_model(model_path , custom_objects={
                    'MultiHeadAttention' : MultiHeadAttention , 
                    'FeedForward' : FeedForward,
                    'correlation_coefficient' : correlation_coefficient,
                    'LayerNormalization' : LayerNormalization,
                    'rc_Conv1D' : rc_Conv1D})
    model._make_predict_function()
    #model.summary()
    session = K.get_session()
    return model , scaler, batch_size,session
   # return model , scaler, batch_size

def population_add_flank(population) : 
    left_flank = ''.join(['T','G','C','A','T','T','T','T','T','T','T','C','A','C','A','T','C'])
    right_flank = ''.join(['G','G','T','T','A','C','G','G','C','T','G','T','T'] )
    population = copy.deepcopy(population)
    for ind in range(len(population)) :
        if not population[ind]!=population[ind]:#math.isnan(population[ind]):       
            population[ind] =  left_flank+ ''.join(population[ind]) + right_flank
        else :
            print(ind)

    return population


def parse_seqs(sequences) :
    sequences = population_add_flank(sequences) ### NOTE : This is different from all other functions ! (User input doesn't have flanks)
    for i in (range(0,len(sequences))) : 
        if (len(sequences[i]) > 110) :
            sequences[i] = sequences[i][-110:]
        if (len(sequences[i]) < 110) : 
            while (len(sequences[i]) < 110) :
                sequences[i] = 'N'+sequences[i]



    A_onehot = np.array([1,0,0,0] ,  dtype=bool)
    C_onehot = np.array([0,1,0,0] ,  dtype=bool)
    G_onehot = np.array([0,0,1,0] ,  dtype=bool) 
    T_onehot = np.array([0,0,0,1] ,  dtype=bool)
    N_onehot = np.array([0,0,0,0] ,  dtype=bool)

    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}
    worddim = len(mapper['A'])
    seqdata = np.asarray(sequences)
    seqdata_transformed = seq2feature(seqdata)


    return np.squeeze(seqdata_transformed) , sequences

class OHCSeq:
    transformed = None
    data = None


def seq2feature(data):
    num_cores = multiprocessing.cpu_count()-2
    nproc = np.min([16,num_cores])
    OHCSeq.data=data
    shared_array_base = mp.Array(ctypes.c_bool, len(data)*len(data[0])*4)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(len(data),len(data[0]),4)
    #OHCSeq.transformed = np.zeros([len(data),len(data[0]),4] , dtype=np.bool )
    OHCSeq.transformed = shared_array


    pool = mp.Pool(nproc)
    r = pool.map(seq2feature_fill, range(len(data)))
    pool.close()
    pool.join()
    #myOHC.clear()
    return( OHCSeq.transformed)





def seq2feature_fill(i):
    mapper = {'A':0,'C':1,'G':2,'T':3,'N':None}
    ###Make sure the length is 110bp
    if (len(OHCSeq.data[i]) > 110) :
        OHCSeq.data[i] = OHCSeq.data[i][-110:]
    elif (len(OHCSeq.data[i]) < 110) : 
        while (len(OHCSeq.data[i]) < 110) :
            OHCSeq.data[i] = 'N'+OHCSeq.data[i]
    for j in range(len(OHCSeq.data[i])):
        OHCSeq.transformed[i][j][mapper[OHCSeq.data[i][j]]]=True 
    return i


def evaluate_model(X,model, scaler, batch_size,session, *graph) :
    #K.set_session(session)
    if(graph) : 
        default_graph = graph[0]

    else : 
        default_graph = tf.get_default_graph()
 
    with default_graph.as_default(): ### attempted to use a closed session is the error i get here.
        with session.as_default() : 
            #K.set_session(session)
            NUM_GPU = len(get_available_gpus())
            if(len(X[0])==80):
                X = population_add_flank(X)
            if( type(X[0])==str or type(X[0])==np.str_) : 
                X = seq2feature(X)
            if NUM_GPU == 0 :    ### Pad for TPU evaluation 
                if(X.shape[0]%batch_size == 0) :
                    Y_pred = model.predict(X , batch_size = batch_size , verbose=1)
                if(X.shape[0]%batch_size != 0) :
                    n_padding = (batch_size*(X.shape[0]//batch_size + 1) - X.shape[0])
                    X_padded = np.concatenate((X,np.repeat(X[0:1,:,:],n_padding,axis=0)))
                    Y_pred_padded = model.predict(X_padded , batch_size = batch_size , verbose=1)
                    Y_pred = Y_pred_padded[:X.shape[0]]
            if NUM_GPU > 0 :    ### Pad for GPU evaluation 
                Y_pred = model.predict(X , batch_size = batch_size , verbose=1)
            Y_pred = [float(x) for x in Y_pred]
            Y_pred = scaler.inverse_transform(Y_pred)
    
    return Y_pred

fitness_function_graph = tf.Graph()
model_condition = 'SC_Ura'
with fitness_function_graph.as_default():
        ## Next line should be a box where you can pick models (on the left side)
    model, scaler,batch_size,session = load_model(model_condition)
    K.set_session(session)

def Ura_surrogate(sequence):
    print("start test!")
    X, _=parse_seqs(sequence)
    # X=tf.cast(X,tf.float32)

    with fitness_function_graph.as_default() : 
        with session.as_default(): 
            #Y_pred= scaler.inverse_transform(model.predict(X,batch_size = 1024,verbose = 0)).flatten()

            Y_pred = evaluate_model(sequence, model, scaler, batch_size ,session, fitness_function_graph)

     #with tf.Session() as sess:
    #   y_res=sess.run(y)
    # output = pd.DataFrame([sequence, Y_pred]).transpose()
    output= np.array(Y_pred)
    return(output)


# with tf.Session() as sess:
    # Initialize variables if needed
#    sess.run(tf.global_variables_initializer())

    # Evaluate and print the values of your tensor
#    tensor_values = sess.run(y)
#print(tensor_values)

