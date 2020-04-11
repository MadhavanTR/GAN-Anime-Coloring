# -*- coding: utf-8 -*-

def define_gan(g_model, d_model, vgg_net1, vgg_net2, learning_rate, pixelLevelLoss_weight=100,\
	       totalVariationLoss_weight=.0001,featureLevelLoss_weight=.01, inp_shape=(512, 512, 3)):
  """
  The function for creating GAN model.
  This function defines the GAN model using the generator and discriminator with 
  discriminator weight updations being freezed during training so that gradients
  flow only to the generator.
  
  So, that discrimnator doesn't dominates over the generator and generator never 
  captures the probability distribution of colored images.
  Parameters:
    g_model (keras.Model): The generator model initialized before.
    d_model (keras.Model): The discriminator model initialized before.
    vgg_net1 (keras.Model): The VGG16 model with layer 4 output initialized for the
      target images.
    vgg_net2 (keras.Model): The VGG16 model with layer 4 output initialized for the 
      generated images.
    learning_rate (float): The learning rate for the model optimizer.
    pixelLevelLoss_weight (float): The weight to be given to pixel level loss.
    totalVariationLoss_weight (float): The weight to be given to total variation loss.
    featureLevelLoss_weight (float): The weight to be given to feature level loss.
    inp_shape (tuple): The input shape for initializing the GAN model.
  Returns:
    tensorflow.keras.Model: The initialized GAN model.
  """
  
  d_model.trainable = False

  # ======= Generator ======= #
  sketch_inp = Input(inp_shape)
  gen_color_output = g_model([sketch_inp])
	
  # ======= Discriminator ======= #
  disc_outputs = d_model([sketch_inp, gen_color_output])
  color_inp = Input(inp_shape)
	
  # =================== PixelLevel Loss =================== #
  pixelLevelLoss = pixelLevel_loss(color_inp, gen_color_output)
  
  # =================== TotalVariation Loss =================== #
  totalVariationLoss = totalVariation_loss(color_inp, gen_color_output)

  # =================== FeatureLevel Loss =================== #  
  net1_outp = vgg_net1([tf.image.resize(color_inp, (224, 224), tf.image.ResizeMethod.BILINEAR)])
  net2_outp = vgg_net2([tf.image.resize(gen_color_output, (224, 224), tf.image.ResizeMethod.BILINEAR)])

  featureLevelLoss = featureLevel_loss(net1_outp, net2_outp)
  
  # =================== Final Model =================== #
  model = Model(inputs=[sketch_inp, color_inp], outputs=disc_outputs)
  
  opt = Adam(lr=learning_rate, beta_1=.5)
	
  # Single output multiple loss functions in keras : https://stackoverflow.com/a/51705573/9079093
  model.compile(loss=lambda y_true, y_pred : tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
                                             pixelLevelLoss_weight * pixelLevelLoss(y_true, y_pred) + \
                                             totalVariationLoss_weight * totalVariationLoss(y_true, y_pred) + \
                                             featureLevelLoss_weight * featureLevelLoss(y_true, y_pred),\
                optimizer=opt)
	
  return model

"""
Creating the generator, discriminator and finally GAN using both.
"""
vgg_net1 = Model(inputs=vgg.input, outputs=ReLU()(vgg.get_layer('block2_conv2').output))
vgg_net2 = Model(inputs=vgg.input, outputs=ReLU()(vgg.get_layer('block2_conv2').output))

g_model = generator(alpha=.2, drop_rate=.5)

d_model = discriminator(alpha=.2, learning_rate=.0002)

gan_model = define_gan(g_model, d_model, vgg_net1, vgg_net2, learning_rate=.0002,\
                       pixelLevelLoss_weight=100, totalVariationLoss_weight=.0001,\
                       featureLevelLoss_weight=.01)