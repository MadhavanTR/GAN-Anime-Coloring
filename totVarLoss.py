# -*- coding: utf-8 -*-
def totalVariation_loss(y, g):
  """
  A loss for smoothness and to remove noise from the output image.
  Custom loss for getting similar colors used in the training data. 
  Parameters:
    y (Tensor): The target images to be generated.
    g (Tensor): The output images by generator.
  
  Returns:
    function: The reference to the loss function of prototype that 
      keras requires.
  """
  import tensorflow.keras.backend as K
  
  def finalTVLoss(y_true, y_pred):
    return K.abs( K.sqrt( K.sum(K.square(g[:, 1:, :, :] - g[:, :-1, :, :])) +\
                          K.sum(K.square(g[:, :, 1:, :] - g[:, :, :-1, :])) ) )
  
  return finalTVLoss