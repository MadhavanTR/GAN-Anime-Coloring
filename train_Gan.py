# -*- coding: utf-8 -*-

def train(g_model, d_model, gan_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs, output_frequency, n_epochs=100, n_batch=128, init_epoch=0):
  """
  A utility function for the training process of GAN.
  The function defines the training the discriminator and the generator alternatively 
  so that gradients flow into only either one of them.
  
  Also prints the loss of discriminator real & generated colored images for every nth batch.
  Parameters:
    g_model (keras.model): The generator model for getting colored during summary. 
    d_model (keras.model): The discriminator model that to be trained.
    gan_model (keras.model): The GAN model that to be trained.
    sketch_paths (numpy.array): The paths to the black-and-white sketches i.e input images.
    image_paths (numpy.array): The paths to the colored images i.e target images.
    latent_dim (int): The dimesnions of latent/random vector(z).
    seed_skets (numpy.array): The fixed black-and-white sketches for checking the generator output after every epoch.
    seed_imgs (numpy.array): The fixed colored images for checking the generator output after every epoch.
    output_frequency (int): The batch frequency at which to print the loss values on the console.  
    n_epochs (int): The # epochs for training the discriminator and GAN.
    n_batch (int): The batch size for every epoch training.
    init_epoch (int): The initial epoch at which to start training process, 
      useful for resuming the training process from a particular epoch.
  """
  
  bat_per_epo = int(TOTAL_IMAGES / n_batch)
  half_batch = int(n_batch / 2)
  
  for i in range(init_epoch, n_epochs):
    start = datetime.now()
    gen_losses = []
    dis_losses = []
    
    for j in range(bat_per_epo):
      # ======================== Train discrimintor on real images ========================= #
      if not j%2:
        X_real_skets, X_real_imgs, y_real = generate_real_samples(sketch_paths, image_paths, half_batch)
        
        d_loss1, _ = d_model.train_on_batch([X_real_skets, X_real_imgs], y_real * .9)
      # ======================== Train discrimintor on real images ========================= #
      
      # ======================== Train discrimintor on generated images ========================= #
        X_fake_skets, X_fake_imgs, y_fake = generate_fake_samples(g_model, sketch_paths, image_paths,\
                                                                  latent_dim, half_batch)

        d_loss2, _ = d_model.train_on_batch([X_fake_skets, X_fake_imgs], y_fake)
      # ======================== Train discrimintor on generated images ========================= #
      d_loss = .5 * (d_loss1 + d_loss2)
      
      # ======================== Train generator on sketch-color images ========================= #
      X_gan_skets, X_gan_imgs, _ = generate_fake_samples(None, sketch_paths, image_paths, latent_dim, n_batch)
      y_gan = np.ones((n_batch, 1))
      
      g_loss = gan_model.train_on_batch([X_gan_skets, X_gan_imgs], y_gan)
      # ======================== Train generator on sketch-color images ========================= #
      
      dis_losses.append(d_loss)
      gen_losses.append(g_loss)
      
      if not j % output_frequency:
        print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

    # Save losses to Tensorboard after every epoch.
    write_log(tensorboard_disc_callback, 'discriminator_loss', np.mean(dis_losses), i+1, (i+1)%3==0)
    write_log(tensorboard_gen_callback, 'generator_loss', np.mean(gen_losses), i+1, (i+1)%3==0)
    
    # Displaying the summary after every epoch.
    display.clear_output(True)
    print('Time for epoch {} : {}'.format(i+1, datetime.now()-start))
    print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    summarize_performance(i, g_model, d_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs, seed_skets.shape[0])
  
  display.clear_output(True)      
  print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
  summarize_performance(i, g_model, d_model, sketch_paths, image_paths, latent_dim, seed_skets, seed_imgs, seed_skets.shape[0])