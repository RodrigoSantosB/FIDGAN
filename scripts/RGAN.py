import pdb
import random
import json
from time import time
from math import floor

import numpy as np
import tensorflow as tf
from scipy.stats import mode

# Local application imports
import data_utils
import plotting
import model
import utils
import eval
import DR_discriminator

# Custom module imports
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio


# Initialize script timing ---------------------------------------------------------------------------------------------
begin = time()

# Disable eager execution for TensorFlow v1 compatibility
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Parse and load settings ----------------------------------------------------------------------------------------------
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())

# If a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']:
    settings = utils.load_settings_from_file(settings)

# Load Data ------------------------------------------------------------------------------------------------------------
data_path = f'./datasets/{settings["data_load_from"]}'
print(f'Loading data from {data_path}')

# Set evaluation flags
settings["eval_an"] = False
settings["eval_single"] = False

samples, labels, index = data_utils.get_data(
    settings["data"], settings["seq_length"], settings["seq_step"],
    settings["num_signals"], settings['sub_id'], settings["eval_single"],
    settings["eval_an"], data_path
)
print(f'samples_size: {samples.shape}')

# Determine the number of variables
num_variables = samples.shape[2]
print(f'num_variables: {num_variables}')

# Save settings and print them 
print('Ready to run with settings:')
for k, v in settings.items():
    print(f'{v} \t {k}')

# Add settings to local environment
locals().update(settings)

# Save settings to a file
settings_path = f'./experiments/settings/{identifier}.txt'
with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=0)


# Build Model ----------------------------------------------------------------------------------------------------------

# 1. Create placeholders for data and model parameters
Z, X, T = model.create_placeholders(batch_size, seq_length, latent_dim, num_variables)

# 2. Extract relevant settings for the discriminator and generator
discriminator_vars = ['hidden_units_d', 'seq_length', 'batch_size', 'batch_mean']
discriminator_settings = {k: settings[k] for k in discriminator_vars}

generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'learn_scale']
generator_settings = {k: settings[k] for k in generator_vars}
generator_settings['num_signals'] = num_variables  # Add number of variables to generator settings

# 3. Define GAN loss functions
D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings)

# 4. Configure optimizers (solvers) for the discriminator and generator
D_solver, G_solver, priv_accountant = model.GAN_solvers(
    D_loss, G_loss, learning_rate, batch_size,
    total_examples=samples.shape[0],
    l2norm_bound=l2norm_bound,
    batches_per_lot=batches_per_lot,
    sigma=dp_sigma, 
    dp=dp
)

# 5. Define the generator output for sample visualization
G_sample = model.generator(Z, **generator_settings, reuse=True)

# # --- evaluation settings--- #
#
# # frequency to do visualisations
# num_samples = samples.shape[0]
# vis_freq = max(6600 // num_samples, 1)
# eval_freq = max(6600// num_samples, 1)
#
# # get heuristic bandwidth for mmd kernel from evaluation samples
# heuristic_sigma_training = median_pairwise_distance(samples)
# best_mmd2_so_far = 1000
#
# # optimise sigma using that (that's t-hat)
# batch_multiplier = 5000 // batch_size
# eval_size = batch_multiplier * batch_size
# eval_eval_size = int(0.2 * eval_size)
# eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
# eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
# n_sigmas = 2
# sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
#     value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
# mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
# with tf.variable_scope("SIGMA_optimizer"):
#     sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
#     # sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
#     # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
# sigma_opt_iter = 2000
# sigma_opt_thresh = 0.001
# sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

# Run the Program ------------------------------------------------------------------------------------------------------

# Configure TensorFlow session to allow GPU memory growth
# tf.random.set_seed(1234)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Initialize TensorFlow session
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())

# Visualize Real Samples

# Select 16 random real samples for visualization
vis_real_indices = np.random.choice(len(samples), size=16, replace=False)
vis_real = np.float32(samples[vis_real_indices, :, :])

# Save plots of real samples
plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=16, num_epochs=num_epochs)
plotting.save_samples_real(vis_real, identifier)

# train ----------------------------------------------------------------------------------------------------------------
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 'latent_dim']
train_settings = dict((k, settings[k]) for k in train_vars)
train_settings['num_signals'] = num_variables

t0 = time()
MMD = np.zeros([num_epochs, ])

# for epoch in range(num_epochs):
for epoch in range(40):
    # -- train epoch -- #
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss, D_solver, G_solver, **train_settings)

    # # -- eval -- #
    # # visualise plots of generated samples, with/without labels
    # # choose which epoch to visualize
    #
    # # random input vectors for the latent space, as the inputs of generator
    # vis_ZZ = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
    #
    # # # -- generate samples-- #
    # vis_sample = sess.run(G_sample, feed_dict={Z: vis_ZZ})
    # # # -- visualize the generated samples -- #
    # plotting.save_plot_sample(vis_sample, epoch, identifier, n_samples=16, num_epochs=None, ncol=4)
    # # plotting.save_plot_sample(vis_sample, 0, identifier + '_real', n_samples=16, num_epochs=num_epochs)
    # # # save the generated samples in cased they might be useful for comparison
    # plotting.save_samples(vis_sample, identifier, epoch)

    print('epoch, D_loss_curr, G_loss_curr, seq_length')
    print('%d\t%.4f\t%.4f\t%d' % (epoch, D_loss_curr, G_loss_curr, seq_length))

    # # -- compute mmd2 and if available, prob density -- #
    # if epoch % eval_freq == 0:
    #     # how many samples to evaluate with?
    #     eval_Z = model.sample_Z(eval_size, seq_length, latent_dim, use_time)
    #     eval_sample = np.empty(shape=(eval_size, seq_length, num_signals))
    #     for i in range(batch_multiplier):
    #         eval_sample[i * batch_size:(i + 1) * batch_size, :, :] = sess.run(G_sample, feed_dict={ Z: eval_Z[i * batch_size:(i + 1) * batch_size]})
    #     eval_sample = np.float32(eval_sample)
    #     eval_real = np.float32(samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier * batch_size), :, :])
    #
    #     eval_eval_real = eval_real[:eval_eval_size]
    #     eval_test_real = eval_real[eval_eval_size:]
    #     eval_eval_sample = eval_sample[:eval_eval_size]
    #     eval_test_sample = eval_sample[eval_eval_size:]
    #
    #     # MMD
    #     # reset ADAM variables
    #     sess.run(tf.initialize_variables(sigma_opt_vars))
    #     sigma_iter = 0
    #     that_change = sigma_opt_thresh * 2
    #     old_that = 0
    #     while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
    #         new_sigma, that_np, _ = sess.run([sigma, that, sigma_solver],
    #                                          feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample})
    #         that_change = np.abs(that_np - old_that)
    #         old_that = that_np
    #         sigma_iter += 1
    #     opt_sigma = sess.run(sigma)
    #     try:
    #         mmd2, that_np = sess.run(mix_rbf_mmd2_and_ratio(eval_test_real, eval_test_sample, biased=False, sigmas=sigma))
    #     except ValueError:
    #         mmd2 = 'NA'
    #         that = 'NA'
    #
    #     MMD[epoch, ] = mmd2

    # -- save model parameters -- #
    model.dump_parameters(sub_id + '_' + str(seq_length) + '_' + str(epoch), sess)

np.save('./experiments/plots/gs/' + identifier + '_' + 'MMD.npy', MMD)

end = time() - begin
print('Training terminated | Training time=%d s' %(end) )

print("Training terminated | training time = %ds  " % (time() - begin))