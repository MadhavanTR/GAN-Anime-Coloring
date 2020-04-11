# -*- coding: utf-8 -*-


def generate_samples(sketch_paths, image_paths, n_samples):
  """
  A function to load black-and-white sketches and colored images.
  The function that loads the black-and-white sketches and corresponding colored images 
  from the given paths for training the GAN.
  Parameters:
    sketch_paths(numpy.array): The paths to the black-and-white sketches i.e input images.
    image_paths(numpy.array): The paths to the colored images i.e target images.
    n_samples(int): The # samples to load for training process.
  Returns:
    X_sketches(numpy.array): The loaded black-and-white sketches.
    X_images(numpy.array): The loaded colored images.
  """

  idxs = np.random.randint(0, TOTAL_IMAGES, n_samples)
  X_sketches = []
  X_images = []
  
  for sket, img in zip(sketch_paths[idxs], image_paths[idxs]):
    X_sketches.append(np.array(Image.open(sket).convert('RGB')))
    X_images.append(np.array(Image.open(img).convert('RGB')))
  
  # Normalizing the values to be between [-1, 1].
  X_sketches = (np.array(X_sketches, dtype='float32')-127.5)/127.5
  X_images = (np.array(X_images, dtype='float32')-127.5)/127.5
	
  return X_sketches, X_images