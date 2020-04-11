# -*- coding: utf-8 -*-
# Reference: https://www.jeremyjordan.me/semantic-segmentation/#loss
def pixelwise_entropy(img_real, img_fake):
    """
    Calulates the per-pixel entropy between target colored and generator colored image.
    Parameters:
      img_real (numpy.array): The target color image. 
      img_fake (numpy.array): The generated color image.
    Returns:
      float: The pixel-wise entropy between generated and target images.
    """
    img_real = (img_real + 1) / 2.
    img_fake = (img_fake + 1) / 2.

    target = np.less_equal(img_real, .5).astype(np.int)
    pred = - np.log(img_fake)

    channel_wise_entropy = np.mean(target * pred, axis=2)
    
    return np.mean(channel_wise_entropy)
  
def pixelwise_accuracy(img_real, img_fake, thresh):
    """
    Calulates the per-pixel accuracy of target colored and generated color image.
    Parameters:
      img_real (numpy.array): The target color image. 
      img_fake (numpy.array): The generated color image.
      thresh (float): The value for thresholding the generated values.
    Returns:
      float: The pixel-wise accuracy between generated and target images.
    """
    img_real = (img_real + 1) / 2.
    img_fake = (img_fake + 1) / 2.

    diffR = np.absolute(np.round(img_real[..., 0]) - np.round(img_fake[..., 0]))
    diffG = np.absolute(np.round(img_real[..., 1]) - np.round(img_fake[..., 1]))
    diffB = np.absolute(np.round(img_real[..., 2]) - np.round(img_fake[..., 2]))

    # Check if the values lie within a threshold.
    predR = np.less_equal(diffR, 1 * thresh)
    predG = np.less_equal(diffG, 1 * thresh)
    predB = np.less_equal(diffB, 1 * thresh)

    # Mutilplying values across the channels.
    pred = predR * predG * predB

    return np.mean(pred)
