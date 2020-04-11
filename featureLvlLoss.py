# -*- coding: utf-8 -*-

def featureLevel_loss(y, g):
  """
  A loss for features extracted from 4th layer of VGG16.
  Custom loss for extracting high level features of the target 
  colored and generated colored images.
  Parameters:
    y (Tensor): The target images to be generated.
    g (Tensor): The output images by the generator.
  
  Returns:
    function: The reference to the loss function of prototype 
      that keras requires.
  """
  import tensorflow.keras.backend as K
  
  def finalFLoss(y_true, y_pred):
    return K.mean( K.sqrt( K.sum( K.square( y - g ) ) ) )
  
  return finalFLoss