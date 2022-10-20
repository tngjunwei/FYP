"""Example / benchmark for building a PTB PSRNN model for character prediction.

To run:

$ python ptb_word_lm.py --data_path=data/

python c:/Users/Carlton/Dropbox/psrnn_tensorflow/psrnn_code/ptb_psrnn_random/ptb_word_lm.py --data_path=c:/Users/Carlton/Dropbox/psrnn_tensorflow/git/psrnn/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

# tf.compat.v1.enable_eager_execution()

if tf.test.gpu_device_name(): 
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
  print("Please install GPU version of TF")

import rnn.psrnn_cell_impl as psrnn_cell_impl
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")

FLAGS = flags.FLAGS

def data_type():
  return tf.float32

class Config(object):
  # Two Stage Regression Parameters
  #2000
  nRFF_Obs = 1000 
  nRFF_P = 1000
  nRFF_F = 1000
  dim_Obs = 20 #20
  dim_P = 20 #20
  dim_F = 20 #20
  reg_rate = 1*10**-3
  obs_window = 20
  kernel_width_Obs = 2 #2
  kernel_width_P = 0.2 #0.2
  kernel_width_F = 0.2 #0.2
    
  # BPTT parameters
  init_scale = 0.0
  learning_rate = 0.5
  max_grad_norm = 0.25
  num_layers = 1
  num_steps = 40  #20
  max_epoch = 100
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 50 #20
  vocab_size = 8  #49
  seed = 0
  hidden_size = dim_Obs


def sequence_loss(logits,
                targets,
                weights,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None,
                name=None):

  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    logits_flat = array_ops.reshape(logits, [-1, num_classes])
    targets_flat = array_ops.reshape(targets, [-1, num_classes])

    crossent = tf.reduce_sum(tf.pow((targets_flat - logits_flat),2), axis=1)

    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    else:
      batch_size = array_ops.shape(logits)[0]
      sequence_length = array_ops.shape(logits)[1]
      crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
    if average_across_timesteps and not average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[1])
      total_size = math_ops.reduce_sum(weights, axis=[1])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, axis=[0])
      total_size = math_ops.reduce_sum(weights, axis=[0])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      crossent /= total_size
    return crossent


class PSRNN(object):
  """The PTB model."""

  def __init__(self, params, config, is_training, gpu_mode=True, reuse=False):
      self.params = params
      self.config = config
      self.is_training = is_training

      with tf.variable_scope('psrnn', reuse=reuse):
        if not gpu_mode:
          with tf.device("/cpu:0"):
            print("model using cpu")
            self.g = tf.Graph()
            with self.g.as_default():
              self.build_model(params, config)
        else:
          print("model using gpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(params, config)
      self.init_session()

  def build_model(self, params, config):

    if self.is_training:
      batch_size = config.batch_size
    else:
      batch_size = 1
    num_steps = config.num_steps
    size = config.hidden_size
    INWIDTH = config.vocab_size
    OUTWIDTH = config.vocab_size

    if self.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def lstm_cell():
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params,
            reuse=tf.get_variable_scope().reuse)
      else:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params)  

    attn_cell = lstm_cell
    if self.is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    psrnns = [attn_cell() for _ in range(config.num_layers)];
    cell = tf.contrib.rnn.MultiRNNCell(
        psrnns, state_is_tuple=True)
  
    self.initial_state = []
    for i in range(config.num_layers):
        self.initial_state.append(tf.constant(np.ones((batch_size,1)).dot(params.q_1.T), dtype=data_type()))
    self.initial_state = tuple(self.initial_state) # (20, 35)

    self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_steps, INWIDTH])
    self.output_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_steps, OUTWIDTH])

    inputs = self.input_x
    targets = self.output_x

    # random fourier features
    W_rff = tf.get_variable('psrnn_W_rff', initializer=tf.constant(params.W_rff.astype(np.float32)), dtype=data_type())
    b_rff = tf.get_variable('psrnn_b_rff', initializer=tf.constant(params.b_rff.astype(np.float32)), dtype=data_type())
    
    self._W_rff = W_rff
    self._b_rff = b_rff

    z = tf.tensordot(tf.cast(inputs, dtype=tf.float32), W_rff,axes=[[2],[0]]) + b_rff
    inputs_rff = tf.cos(z)*np.sqrt(2.)/np.sqrt(config.nRFF_Obs)

    # dimensionality reduction
    U = tf.get_variable('psrnn_U', initializer=tf.constant(params.U.astype(np.float32)),dtype=data_type())
    U_bias = tf.get_variable('psrnn_U_bias',[config.hidden_size],initializer=tf.constant_initializer(0.0))

    inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2],[0]]) + U_bias

    # update rnn state
    inputs_unstacked = tf.unstack(inputs_embed, num=num_steps, axis=1) 

    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs_unstacked, initial_state=self.initial_state)

    # reshape output
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    
    initializer = tf.constant(params.W_pred.T.astype(np.float32))
    softmax_w = tf.get_variable("softmax_w", initializer=initializer, dtype=data_type())
    initializer = tf.constant(params.b_pred.T.astype(np.float32))
    softmax_b = tf.get_variable("softmax_b", initializer=initializer, dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    self.final_state = state

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, INWIDTH])

    loss = sequence_loss(
        logits,
        targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )
    
    # one_step_pred = tf.argmax(logits,axis=2)
    
    self._cost = self.cost = tf.reduce_mean(loss)
    
    # initialize vars
    self.init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    self.assign_ops = {}
    for var in tvars:
      pshape = var.get_shape()
      pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
      assign_op = var.assign(pl)
      self.assign_ops[var] = (assign_op, pl)

    if not self.is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)

  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()

  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        if 'psrnn' in param_name:
          p = self.sess.run(var)
          model_names.append(param_name)
          params = np.round(p*10000).astype(np.int).tolist()
          model_params.append(params)
          model_shapes.append(p.shape)
    return model_params, model_shapes, model_names

  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam

  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)

  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if 'psrnn' in var.name:
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          #print(f"pshape: {pshape}, p.shape: {p.shape}")
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.assign_ops[var]
          self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1

  def load_json(self, no_init=False, jsonfile=None):
    self.init_psrnn_weights(no_init)

  def save_json(self, jsonfile='weights_psrnn.json'):
    self.save_psrnn_weights() # decoder weights are fixed

  def save_psrnn_weights(self):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open('./weights_psrnn.json', 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  def init_psrnn_weights(self, no_init=False):
    if no_init or not os.path.exists('./weights_psrnn.json'):
      return

    with open("./weights_psrnn.json", 'r') as f:
      params = json.load(f)

    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if 'psrnn' in var.name:
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          #print(f"pshape: {pshape}, p.shape: {p.shape}")
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.assign_ops[var]
          self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1

def rnn_init_state(rnn):
    return rnn.sess.run(rnn.initial_state)

def rnn_next_state(rnn, z, a, prev_state):
  #input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 3))), axis=2)
  input_x = z.reshape((1,Config.num_steps,Config.vocab_size))
  feed = {rnn.input_x: input_x, rnn.initial_state:prev_state}
  return rnn.sess.run(rnn.final_state, feed)

def rnn_output_size(mode):
  # if mode == MODE_ZCH:
  #   return (32+256+256)
  # if (mode == MODE_ZC) or (mode == MODE_ZH):
  #   return (32+256)
  return Config.vocab_size # MODE_Z or MODE_Z_HIDDEN #32

def rnn_output(state, z, mode):
  # if mode == MODE_ZCH:
  #   return np.concatenate([z, np.concatenate((state.c,state.h), axis=1)[0]])
  # if mode == MODE_ZC:
  #   return np.concatenate([z, state.c[0]])
  # if mode == MODE_ZH:
  #   return np.concatenate([z, state.h[0]])
  return z[-1] # MODE_Z or MODE_Z_HIDDEN
