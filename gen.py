# -*- coding: utf-8 -*-


def generator(drop_rate, alpha, inp_shape=(512, 512, 3)):
  """
  A function for creating generator model.
  This function defines the generator part of the GAN with given input shape 
  using the Convolution and Convolution Transpose Blocks defined before.
  
  Takes the sketch input with values in the range [-1, 1] and generates the
  colored image of the same.
  Parameters:
    drop_rate (float): The drop out rate for regularizarition.
    alpha (float): The alpha value for LeakyReLu activation.
    inp_shape (tuple): The shape of input for initializing generator.
  
  Returns:
    tensorflow.keras.Model: The generator model initialized with a U-Net structure.
  """
  
  n_filters = 16
  
  inp = Input(inp_shape)

  print('Encoder:')
  conv1 = convBlock(inp, n_filters, BN=False, alpha=alpha)# 256x256
  conv2 = convBlock(conv1, n_filters*2, alpha=alpha)      # 128x128
  conv3 = convBlock(conv2, n_filters*4, alpha=alpha)      # 64x64
  conv4 = convBlock(conv3, n_filters*8, alpha=alpha)      # 32x32
  conv5 = convBlock(conv4, n_filters*8, alpha=alpha)      # 16x16
  conv6 = convBlock(conv5, n_filters*8, alpha=alpha)      # 8x8
  conv7 = convBlock(conv6, n_filters*8, alpha=alpha)      # 4x4
  conv8 = convBlock(conv7, n_filters*8, alpha=alpha)      # 2x2x512

  print('Decoder:')
  deconv1 = convTransBlock(conv8, n_filters*8, alpha=alpha)                                     # 4x4
  deconv2 = convTransBlock(deconv1, n_filters*8, convOut=conv7, dropout=drop_rate, alpha=alpha) # 8x8
  deconv3 = convTransBlock(deconv2, n_filters*8, convOut=conv6, dropout=drop_rate, alpha=alpha) # 16x16
  deconv4 = convTransBlock(deconv3, n_filters*8, convOut=conv5, dropout=drop_rate, alpha=alpha) # 32x32
  deconv5 = convTransBlock(deconv4, n_filters*4, convOut=conv4, alpha=alpha)                    # 64x64
  deconv6 = convTransBlock(deconv5, n_filters*2, convOut=conv3, alpha=alpha)                    # 128x128
  deconv7 = convTransBlock(deconv6, n_filters, convOut=conv2, alpha=alpha)                      # 256x256
  deconv8 = convTransBlock(deconv7, 3, convOut=conv1, activation=False, BN=False)               # 512x512

  outp = tanh(deconv8)

  model = Model(inputs=inp, outputs=outp)

  return model