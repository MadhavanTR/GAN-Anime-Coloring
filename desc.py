# -*- coding: utf-8 -*-

def discriminator(alpha, learning_rate, inp_shape=(512, 512, 3), target_shape=(512, 512, 3)):
  """
  A function for creating discriminator model.
  This function defines the discriminator part of the GAN with given input 
  shape & target shape using the Convolution Blocks defined before.
  
  Takes the sketch and target image/generated colored image from generator 
  with the values in the range [-1, 1] and outputs the probability of being 
  real/fake in the range [0, 1].
  Parameters:
    alpha (float): The alpha value for LeakyReLu activation.
    learning_rate (float): The learning rate value for discriminator optimizer.
    inp_shape (tuple): The shape of input for initializing generator.
    target_shape (tuple): The shape of the target output image. 
  
  Returns:
    tensorflow.keras.Model: The initialized discriminator model.
  """
  
  n_filters = 16
  
  inp1 = Input(inp_shape) # sketch input
  inp2 = Input(target_shape) # colored input

  inp = concatenate([inp1, inp2])                                 # 512x512
  conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha)        # 256x256
  conv2 = convBlock(conv1, n_filters*2, alpha=alpha)              # 128x128
  conv3 = convBlock(conv2, n_filters*4, alpha=alpha)              # 64x64
  conv4 = convBlock(conv3, n_filters*8, alpha=alpha)              # 32x32
  conv5 = convBlock(conv4, n_filters*8, filter_size=2, stride=1,\
                    padding='valid', alpha=alpha)                 # 31x31x512
  conv6 = convBlock(conv5, n_filters=1, filter_size=2, stride=1,\
                    activation=False, BN=False, padding='valid')  # 30x30x1

  sigmoid_outp = sigmoid(conv6)
  
  outp = GlobalAveragePooling2D()(sigmoid_outp)

  model = Model(inputs=[inp1, inp2], outputs=outp)

  opt = Adam(lr=learning_rate, beta_1=.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  
  return model