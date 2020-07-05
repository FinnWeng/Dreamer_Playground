import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools


class RSSM(tools.Module):
  '''
  recurrent state space model
  '''

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype)) # zero initialization, float32

  @tf.function
  def observe(self, embed, action, state=None):
    # embed  (25, 50, 1024)
    # action (25, 50, 4)
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2]) # => embed  (50, 25, 1024)
    action = tf.transpose(action, [1, 0, 2]) # => action (50, 25, 4)
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None): # been used only for summary
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state): 
    '''
    which means:
    some of features are stochastic, some are deterministic
    stoch is sampled from post of observation step of imagination
    deterministic is from RSSM's GRUcell
    '''
    return tf.concat([state['stoch'], state['deter']], -1) 

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    '''
    p(st|st-1,at-1,ot)
    '''
    prior = self.img_step(prev_state, prev_action) # get the prior of VAE

    # below gets posterior of VAE
    x = tf.concat([prior['deter'], embed], -1) # concat state and observation
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    '''
    p(st|st-1,at-1)
    '''
    # the image step from St-1 to St (be aware that this is not OBSERVATION)
    x = tf.concat([prev_state['stoch'], prev_action], -1) # => (25, 34)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']]) # (output, next_state) = call(input, state)
    # x: (25, 200)
    deter = deter[0]  # Keras wraps the state in a list.

    # below is VAE prior, which means P(z|x)
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x) # =>(25, 60)

  
    mean, std = tf.split(x, 2, -1) # =>(25, 30), =>(25, 30) or =>(1250, 30), =>(1250, 30), the 1250 is 25*50 since the flatten function of imagine head
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    print("stoch:",stoch.shape) # (25, 30)
    
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior
  


  @tf.function
  def img_step_and_get_action(self, prev_state, prev_action):
    '''
    p(st|st-1,at-1)
    '''
    # the image step from St-1 to St (be aware that this is not OBSERVATION)
    x = tf.concat([prev_state['stoch'], prev_action], -1) # => (25, 34)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']]) # (output, next_state) = call(input, state)
    # x: (25, 200)
    deter = deter[0]  # Keras wraps the state in a list.

    # below is VAE prior, which means P(z|x)
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x) # =>(25, 60)

  
    mean, std = tf.split(x, 2, -1) # =>(25, 30), =>(25, 30) or =>(1250, 30), =>(1250, 30), the 1250 is 25*50 since the flatten function of imagine head
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    print("stoch:",stoch.shape) # (25, 30)
    
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return (prior, prev_action)


  

class ConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    # print("obs['image'].shape:",obs['image'].shape) #  (25, 50, 64, 64, 3)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:])) # (1250, 64, 64, 3)
    x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x) # (1250, 31, 31, 32)
    x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x) # (1250, 14, 14, 64)
    x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x) # (1250, 6, 6, 128)
    x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x) #  (1250, 2, 2, 256)
    
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0) # 8*2*2 = 32
   
    return tf.reshape(x, shape) # (25, 50, 1024)


class ConvDecoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
    self._act = act
    self._depth = depth
    self._shape = shape

  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseDecoder(tools.Module):

  def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
    # (), 3, self._c.num_units = 400, act=act
    self._shape = shape
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act

  def __call__(self, features):
    x = features
    for index in range(self._layers): # 3
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
      # print("x:",x.shape) #  (15, 1250, 400)
    x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
    # print("x:",x.shape)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    # print("x:",x.shape)
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
    raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

  def __init__(
      self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
      min_std=1e-4, init_std=5, mean_scale=5):
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  def __call__(self, features):
    raw_init_std = np.log(np.exp(self._init_std) - 1)
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      mean, std = tf.split(x, 2, -1)
      mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
      std = tf.nn.softplus(std + raw_init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      dist = tools.OneHotDist(x)
    else:
      raise NotImplementedError(dist)
    return dist
