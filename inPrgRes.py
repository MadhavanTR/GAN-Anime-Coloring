# -*- coding: utf-8 -*-
def save_plot(predictions, epoch, n=3):
  """
  A function for saving intermediate predictions.
  The function saves plot of the generated color images by the generator 
  for seed/fixed sketches that are loaded before we start training.
  Parameters:
    predictions (numpy.array): The colored images by the generator.
    epoch (int): The epoch at which the colored images are generated.
    n (int): The number of colored images generated.
  """
  n = int(math.sqrt(n))
  plt.figure(figsize=(6, 6))
  
  # Rescaling back into the range of [0, 255] from [-1, 1].
  predictions = (predictions + 1) / 2.0
  
  for i in range(n * n):
    plt.subplot(n, n, 1 + i)
    plt.axis('off')
    plt.imshow(predictions[i])
  
  filename = './Sketch2Image/generated_plot_e%03d.png' % (epoch+1)
  plt.savefig(filename)
  plt.show()