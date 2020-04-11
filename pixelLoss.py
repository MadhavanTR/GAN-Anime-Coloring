# -*- coding: utf-8 -*-

def pixelLevel_loss(y, g):
  """
  A loss for getting proper images by comparing each pixel.
  Custom loss for Pixel2Pixel level translation so that colors don't 
  come out the edges of generated images.
  Parameters:
    y (Tensor): The real target images to be generated.
    g (Tensor): The output images by the generator.
  
  Returns:
    function: The reference to the loss function of the prototype 
      that keras requires.
  """
  import tensorflow.keras.backend as K
  
  def finalPLLoss(y_true, y_pred):
    return K.mean( K.abs( y - g ) )
  
  return finalPLLoss