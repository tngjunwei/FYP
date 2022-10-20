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
import cv2
import time

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

import reader
import psrnn_cell_impl
from rnn.two_stage_regression import get_params
import os
# from vae import ConvVAE

# vae = ConvVAE()
# vae.load_json()

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")

FLAGS = flags.FLAGS
losses = []

g = tf.Graph()

class Config(object):
  # Two Stage Regression Parameters
  #2000
  nRFF_Obs = 2000 
  nRFF_P = 2000
  nRFF_F = 2000
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
  vocab_size = 32  #49
  seed = 0
  hidden_size = dim_Obs


# input shape = (batch_size, num_steps, 64, 64, 3)
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


def data_type():
  return tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    print("EPOCH SIZE: ", self.epoch_size)
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, config.vocab_size, name=name)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, params):
    self._input = input_
    self.config = config
    self.is_training = is_training

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

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
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    psrnns = [attn_cell() for _ in range(config.num_layers)];
    cell = tf.contrib.rnn.MultiRNNCell(
        psrnns, state_is_tuple=True)
  
    self._initial_state = []
    for i in range(config.num_layers):
        self._initial_state.append(tf.constant(np.ones((batch_size,1)).dot(params.q_1.T), dtype=data_type()))
    self._initial_state = tuple(self._initial_state) # (20, 35)

    inputs = input_.input_data
    print("Inputs: ", inputs.shape) #(20, 20, 32)

    # random fourier features
    W_rff = tf.get_variable('psrnn_W_rff', initializer=tf.constant(params.W_rff.astype(np.float32)), dtype=data_type())
    b_rff = tf.get_variable('psrnn_b_rff', initializer=tf.constant(params.b_rff.astype(np.float32)), dtype=data_type())
    
    self._W_rff = W_rff
    self._b_rff = b_rff

    print("W_rff: ", W_rff.shape)
    print("b_rff: ", b_rff.shape)

    z = tf.tensordot(tf.cast(inputs, dtype=tf.float32), W_rff,axes=[[2],[0]]) + b_rff
    inputs_rff = tf.cos(z)*np.sqrt(2.)/np.sqrt(config.nRFF_Obs)

    print("inputs_rff: ", inputs_rff.shape)
    print("z: ", z.shape)

    # dimensionality reduction
    U = tf.get_variable('psrnn_U', initializer=tf.constant(params.U.astype(np.float32)),dtype=data_type())
    U_bias = tf.get_variable('psrnn_U_bias',[config.hidden_size],initializer=tf.constant_initializer(0.0))

    print("U: ", U.shape)

    inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2],[0]]) + U_bias

    
    print("inputs_embed: ", inputs_embed.shape)

    # update rnn state
    inputs_unstacked = tf.unstack(inputs_embed, num=num_steps, axis=1) 

    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs_unstacked, initial_state=self._initial_state)

    # reshape output
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    

    initializer = tf.constant(params.W_pred.T.astype(np.float32))
    softmax_w = tf.get_variable("softmax_w", initializer=initializer, dtype=data_type())
    initializer = tf.constant(params.b_pred.T.astype(np.float32))
    softmax_b = tf.get_variable("softmax_b", initializer=initializer, dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b


    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
    print("Logit shape: ", logits.shape)
    self.inputs = inputs
    self.results = logits

    loss = sequence_loss(
        logits,
        input_.targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )

    print("Loss: ", loss.shape) #(20,)
    

    self._cost = cost = tf.reduce_mean(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})    

  def init_psrnn_weights(self, session):
    with open("./weights_psrnn.json", 'r') as f:
      params = json.load(f)

    with g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if 'psrnn' in var.name:
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.assign_ops[var]
          session.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1

  def get_model_params(self, sess, filter=""):
  # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        if filter in var.name:
          param_name = var.name
          p = sess.run(var)
          model_names.append(param_name)
          params = np.round(p*10000).astype(np.int).tolist()
          model_params.append(params)
          model_shapes.append(p.shape)
    
    return model_params, model_shapes, model_names

  def save_weights(self, session, filter=""):
    model_params, model_shapes, model_names = self.get_model_params(session, "psrnn")
    qparams = []
    for p in model_params:
      qparams.append(p)
    
    jsonfile = f'weights_psrnn_{int(time.time())}.json'
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))


  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  # @property
  # def num_correct_pred(self):
  #     return self._num_correct_pred

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  
def run_epoch(session, model, eval_op=None, verbose=False, save_params=False, epoch=0):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  correct_pred = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      # "num_correct_pred": model.num_correct_pred,
      "final_state": model.final_state
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
      
    feed_dict = {}
    for i, s in enumerate(model.initial_state):
      feed_dict[s] = state[i]

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps
    # correct_pred += vals["num_correct_pred"]
    
    perplexity = np.exp(costs / iters)
    bpc = np.log2(perplexity)
    # accuracy = correct_pred/iters
    accuracy = 0

  return perplexity, bpc, accuracy, np.mean(cost)


def main(_):


  raw_data = np.load(f"./series_{Config.vocab_size}.npz")
  data_mu = raw_data["mu"]
  data_logvar = raw_data["logvar"]
  data_action =  raw_data["action"]


  mu = data_mu
  logvar = data_logvar
  raw_a = data_action

  s = logvar.shape
  raw_z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  
  print("Raw z shape: ", raw_z.shape)
  inputs = raw_z[:, :, 0:Config.vocab_size]

  print("Before reshape: ", inputs.shape) 
  inputs = np.reshape(inputs, (-1, Config.vocab_size))
  print("Shape of training inputs: ", inputs.shape) 


  train_size = int(0.8 * inputs.shape[0])
  val_size = inputs.shape[0] - train_size
  two_stage_size = int(train_size)


  train_data = inputs[:train_size]
  valid_data = inputs[train_size:train_size+val_size]
  two_stage_data = train_data[:two_stage_size]

  print("Train shape: ", train_data.shape)
  print("Val shape: ", valid_data.shape)

  config = Config()
  eval_config = Config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  # print the config
  for i in inspect.getmembers(config):
    # Ignores anything starting with underscore 
    # (that is, private and protected attributes)
    if not i[0].startswith('_'):
        # Ignores methods
        if not inspect.ismethod(i[1]):
            print(i)
            

  # perform two stage regression to obtain initialization for PSRNN
  params = get_params(two_stage_data, Config(), no_init=True)


  with g.as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
                                                
    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, params=params)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, params=params)
      tf.summary.scalar("Validation Loss", mvalid.cost)


    sv = tf.train.Supervisor()
    with sv.managed_session() as session:

      valid_perplexity_all = []        
      valid_bpc_all = []
      valid_acc_all = []

      valid_perplexity, valid_bpc, valid_acc, cost = run_epoch(session, mvalid)
      print("train_cost, val_cost")
      valid_perplexity_all.append(valid_perplexity)
      valid_bpc_all.append(valid_bpc)
      valid_acc_all.append(valid_acc)
      
      for i in range(config.max_epoch):
        m.assign_lr(session, config.learning_rate * config.lr_decay)
        
        train_perplexity, train_bpc, train_acc, train_cost = run_epoch(session, m, eval_op=m.train_op, verbose=True, epoch=i)
        
        valid_perplexity, valid_bpc, valid_acc, val_cost = run_epoch(session, mvalid, epoch=-i)
        print(f"{train_cost},{val_cost}")
        valid_perplexity_all.append(valid_perplexity)
        valid_bpc_all.append(valid_bpc)
        valid_acc_all.append(valid_acc)
    
      m.save_weights(session, filter="psrnn") # save psrnn weights at the last step

if __name__ == "__main__":
  tf.app.run()
