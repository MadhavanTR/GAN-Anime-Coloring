# -*- coding: utf-8 -*-


"""
Load the TensorBoard notebook extension.
"""
%load_ext tensorboard

"""
Tensorboard callbacks for logging generator and discriminator loss.
"""
tensorboard_gen_callback = TensorBoard(log_dir="logs/generator/")
tensorboard_gen_callback.set_model(g_model)

tensorboard_disc_callback = TensorBoard(log_dir="logs/discriminator/")
tensorboard_disc_callback.set_model(d_model)

"""
Launching the tensorboard.
"""
%tensorboard --logdir logs