import json
import os
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
# import tensorflow_addons as tfa

# Local application imports
import data_utils
import utils
import autoencoderFunctions
import mmd
from mod_core_rnn_cell_impl import LSTMCell


# Initialize script timing ---------------------------------------------------------------------------------------------
begin = time()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Parse and load settings ----------------------------------------------------------------------------------------------
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# Load Data ------------------------------------------------------------------------------------------------------------
data_path = f'./datasets/{settings["data_load_from"]}'
print('-' * 100)
print(f'Loading data from {data_path}')

# Set evaluation flags
settings["eval_an"] = False
settings["eval_single"] = False
# samples, pdf, labels = data_utils.get_data(settings)

samples, labels, index = data_utils.get_data(
    settings["data"], settings["seq_length"], settings["seq_step"],
    settings["num_signals_autoencoder"], settings['sub_id'], settings["eval_single"],
    settings["eval_an"], data_path
)

# Determine the number of variables
num_variables = samples.shape[2]
print(f'num_variables: {num_variables}')

# Save settings and print them 
print('Ready to run with settings:')
for k, v in settings.items():
    print(f'{v} \t {k}')
print('-' * 100)

# Locally defined parameters and paths ---------------------------------------------------------------------------------
identifier = settings["identifier"]
sub_id = settings["sub_id"]
seq_length = settings["seq_length"]
hidden_units_g = settings["hidden_units_g"]
num_generated_features = settings["num_generated_features"]

batch_size = settings["batch_size_autoencoder"]
seq_length = settings["seq_length_autoencoder"]
num_signals = settings["num_signals_autoencoder"]
latent_dim = settings["latent_dim_autoencoder"]
hidden_units = settings["hidden_units_autoencoder"]
learning_rate = settings["learning_rate_autoencoder"]
training_epochs = settings["training_epochs_autoencoder"]
display_step = settings["display_step_autoencoder"]
path_autoencoder_training_parameters = settings["path_autoencoder_training_parameters"]
path_autoencoder_training_results = settings["path_autoencoder_training_results"]
generatorEpoch = settings["generatorEpoch"]  # epoch of the generator to be used (TODO change to get the best epoch)

# Save settings to a file
settings_path = f'./experiments/settings/{identifier}.txt'
with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=0)

# Create Encoder Model -------------------------------------------------------------------------------------------------
X = tf.compat.v1.placeholder(tf.float32, [batch_size, seq_length, num_signals])
z_enc_outputs = autoencoderFunctions.encoderModel(
    X, hidden_units, seq_length, batch_size, latent_dim, reuse=False, parameters=None
)

# Load Pre Trained Generator Model -------------------------------------------------------------------------------------
para_path = './experiments/parameters/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}'+ '/' + sub_id + '_' + str(generatorEpoch) + '.npy'
parameters = autoencoderFunctions.loadParameters(para_path)
x_dec_outputs, x_dec_outputs_l = autoencoderFunctions.generatorModel(
    z_enc_outputs, hidden_units_g, seq_length, 
    batch_size, num_generated_features, reuse=False, parameters=parameters
)

# Get true and prediction outputs --------------------------------------------------------------------------------------
encoder_inputs = [tf.compat.v1.reshape(X, [-1, num_signals])]
encoder_outputs = [tf.compat.v1.reshape(z_enc_outputs, [-1, latent_dim])]
decoder_outputs = [tf.compat.v1.reshape(x_dec_outputs, [-1, num_signals])]
# decoder_outputs = [tf.compat.v1.reshape(x_dec_outputs_l, [-1, num_signals])]

y_true = [tf.compat.v1.reshape(yt, [-1]) for yt in encoder_inputs]
y_encoded = [tf.compat.v1.reshape(ye, [-1]) for ye in encoder_outputs]
y_pred = [tf.compat.v1.reshape(yp, [-1]) for yp in decoder_outputs]

# Load the trained Discriminator and Evaluate its value for x and G(E(x)) ----------------------------------------------
# d_output_xr, d_logits_xr = autoencoderFunctions.discriminatorModel(X, settings['hidden_units_d'], reuse=False, parameters=parameters)
# d_output_xg, d_logits_xg = autoencoderFunctions.discriminatorModelPred(x_dec_outputs, settings['hidden_units_d'], reuse=False, parameters=parameters)

# disc_outputs_x = [tf.reshape(d_output_xr, [-1, num_signals])]
# disc_outputs_xg = [tf.reshape(d_output_xg, [-1, num_signals])]

# d_y_true = [tf.reshape(dyt, [-1]) for dyt in disc_outputs_x]
# d_y_pred = [tf.reshape(dyp, [-1]) for dyp in disc_outputs_xg]

# Define Trainable Variables -------------------------------------------------------------------------------------------
t_vars = tf.compat.v1.trainable_variables()
# train_vars = [var for var in t_vars]    # trains all variables

# trains the encoder variables
train_vars = tf.compat.v1.get_collection(
    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "encoder"
)   

# trains the generator variables
train_vars = train_vars + tf.compat.v1.get_collection(
    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "generator"
)   
print("train_vars: ", train_vars)

# Define Loss and Optimizer --------------------------------------------------------------------------------------------
loss = 0
for i in range(len(y_true)):
    loss += tf.compat.v1.reduce_sum(
        tf.compat.v1.square(tf.compat.v1.subtract(y_pred[i], y_true[i]))
    )   # loss used in first results
   
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.1).minimize(loss, var_list=train_vars)

# Initialize variables and launch graph --------------------------------------------------------------------------------
init = tf.compat.v1.initialize_all_variables()
l = []
a_values = []
b_values = []
c_values = []
os.makedirs(path_autoencoder_training_parameters + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}', exist_ok=True)
os.makedirs(path_autoencoder_training_results + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}', exist_ok=True)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        loss_epoch = 0
        num_batches = int(samples.shape[0]/batch_size)
        for batch in range(num_batches):    # for batch in range(1):
            # i = batch*500
            # x = samples[i:i + 500]
            i = batch*batch_size
            x = samples[i:i + batch_size]
            x = x.reshape((batch_size, seq_length, num_signals))
            feed = {X: x}
            # Fit training using batch data
            _, loss_batch = sess.run([optimizer, loss], feed_dict=feed)
            loss_epoch = loss_epoch + np.sum(loss_batch)

        l = np.append(l, loss_epoch)
        
        if epoch % display_step == 0:
            a, b, c = sess.run([y_pred, y_true, y_encoded], feed_dict=feed)
            a_values = np.append(a_values, a)
            b_values = np.append(a_values, b)
            c_values = np.append(a_values, c)
            print('-' * 100)
            # print ("logits")
            # print (a)
            # print ("labels")
            # print (b)
            # print ("encoded")
            # print (c)
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(loss_epoch))

        # if (epoch == (training_epochs-1)):
            # print("train_vars: ", train_vars)
            # Modificar o nome do arquivo ou pasta para salvar outros resultados
            # autoencoderFunctions.dumpParameters(path_autoencoder_training_parameters + str(epoch), sess)
            autoencoderFunctions.dumpParameters(path_autoencoder_training_parameters + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}' + '/' + sub_id + '_' + str(epoch), sess)

    np.savetxt(path_autoencoder_training_results + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}' + '/' + 'loss_per_epoch.txt', l, fmt='%f')   
    # Modificar o nome do arquivo ou pasta para salvar outros resultados
    np.savetxt(path_autoencoder_training_results + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}' + '/' + 'a_values.txt', a_values, fmt='%f')  
    # Modificar o nome do arquivo ou pasta para salvar outros resultados
    np.savetxt(path_autoencoder_training_results + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}' + '/' + 'b_values.txt', b_values, fmt='%f')  
    # Modificar o nome do arquivo ou pasta para salvar outros resultados
    np.savetxt(path_autoencoder_training_results + '/' + f'{identifier}_{seq_length}_{num_signals}_{latent_dim}' + '/' + 'c_values.txt', c_values, fmt='%f')

    elapsedTime = time() - begin
    print('-' * 100)
    print(" Optimization Finished! elapsedTime: ", elapsedTime)
