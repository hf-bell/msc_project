from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from tensorflow.keras.activations import softplus
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
import numpy as np
tf.compat.v1.experimental.output_all_intermediates(True)

def metric_orth_dist(true_position, pred_position):
    yaw_true = (true_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
    # Finally compute orthodromic distance
    delta_long = tf.abs(tf.atan2(tf.sin(yaw_true - yaw_pred), tf.cos(yaw_true - yaw_pred)))
    numerator = tf.sqrt(tf.pow(tf.cos(pitch_pred)*tf.sin(delta_long), 2.0) + tf.pow(tf.cos(pitch_true)*tf.sin(pitch_pred)-tf.sin(pitch_true)*tf.cos(pitch_pred)*tf.cos(delta_long), 2.0))
    denominator = tf.sin(pitch_true)*tf.sin(pitch_pred)+tf.cos(pitch_true)*tf.cos(pitch_pred)*tf.cos(delta_long)
    great_circle_distance = tf.abs(tf.atan2(numerator, denominator))
    return great_circle_distance

# This way we ensure that the network learns to predict the delta angle
def toPosition(values):
    orientation = values[0]
    magnitudes = values[1]/2.0
    directions = values[2]
    # The network returns values between 0 and 1, we force it to be between -2/5 and 2/5
    motion = magnitudes * directions

    yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

    cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
    cond_correct = tf.cast(tf.logical_and(tf.less_equal(pitch_pred_wo_corr, 1.0), tf.greater_equal(pitch_pred_wo_corr, 0.0)), tf.float32)
    cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

    pitch_pred = cond_above * (1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (-pitch_pred_wo_corr)
    yaw_pred = tf.math.mod(cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below * (yaw_pred_wo_corr - 0.5),1.0)
    return tf.concat([yaw_pred, pitch_pred], -1)

# ----------------------------- TRAIN ----------------------------

def create_pos_only_model_7(M_WINDOW, H_WINDOW, BETA_h=1e-5, BETA_c=1e-6):
    # Defining model structure
    encoder_inputs = Input(shape=(M_WINDOW, 2))
    decoder_inputs = Input(shape=(1, 2))

    lstm_layer_enc = LSTM(1024, return_sequences=True, return_state=True)
    lstm_layer_dec = LSTM(512, return_sequences=True, return_state=True)
    h_state_mapper = Dense(1024, activation='linear', name='h_state_dist')
    c_state_mapper = Dense(1024, activation='linear', name='c_state_dist')
    decoder_dense_mot = Dense(4, activation='sigmoid')
    decoder_dense_dir = Dense(4, activation='tanh')
    To_Position = Lambda(toPosition)

    # Encoding
    encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_inputs)
    state_h = h_state_mapper(state_h)
    state_c = c_state_mapper(state_c)
    mu_h, sigma_h = state_h[:, :512], state_h[:, 512:]
    mu_c, sigma_c = state_c[:, :512], state_c[:, 512:]
    state_h_distribution = tfp.distributions.Normal(mu_h, softplus(sigma_h - 5))
    state_h = state_h_distribution.sample()
    state_h_sample = state_h
    state_c_distribution = tfp.distributions.Normal(mu_c, softplus(sigma_c - 5))
    state_c = state_c_distribution.sample()
    state_c_sample = state_c
    states = [state_h, state_c]

    # Decoding
    all_outputs = []
    inputs = decoder_inputs
    outs_distributions = []
    for curr_idx in range(H_WINDOW):
        # # Run the decoder on one timestep
        decoder_pred, state_h, state_c = lstm_layer_dec(inputs, initial_state=states)
        outputs_delta = decoder_dense_mot(decoder_pred)
        outputs_delta_dir = decoder_dense_dir(decoder_pred)
        outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs_pos)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = outputs_pos
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    # decoder_outputs = all_outputs

    # BETA_h = 1e-5
    # BETA_c = 1e-6
    marginal_h = marginal_tril_dist('h')
    marginal_c = marginal_tril_dist('c')


    def vib_loss(y_true, y_pred):
        class_loss = metric_orth_dist(y_true, y_pred)
        info_loss_h = state_h_distribution.log_prob(state_h_sample)
        info_loss_c = state_c_distribution.log_prob(state_c_sample)
        marginal_loss_h = marginal_h.log_prob(tf.cast(state_h_sample, tf.float32))
        marginal_loss_c = marginal_c.log_prob(tf.cast(state_c_sample, tf.float32))
        return class_loss + BETA_h * (info_loss_h - marginal_loss_h) + BETA_c * (info_loss_c - marginal_loss_c) 

    # Define and compile model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    opt = Adam(5e-4)
    model.compile(optimizer=opt, loss=vib_loss, metrics=[metric_orth_dist, vib_loss])
    return model


# Define learnable marginal distribution for UQ
def marginal_tril_dist(h_or_c='h'):
    n = 200
    d = 512
    min_variance = 1e-5
    # Compute the number of parameters needed for the lower triangular covariance matrix
    tril_components = (d  * (d + 1)) // 2

    # Parameterize the categorical distribution for the mixture
    #mix_logits = tf.Variable('mix_logits', [n]) # get or create params to fill marginal dist.
    init_vals = np.random.rand(n)
    init_probs = [float(i)/sum(init_vals) for i in init_vals]
    mix_logits = tf.cast(np.log(init_probs), dtype=tf.float32)
    mix_logits = trainable_dist_layer(mix_logits, name='mix_logits_%s'%(h_or_c))
    #mix_logits = tf.Variable(initial_value=np.log(init_probs), name='mix_logits', dtype=tf.float32)

    mix_dist = tfp.distributions.Categorical(logits=mix_logits.return_vars())

    # Parameterize the means of Gaussian distribution
    mu_init = tf.initializers.RandomNormal()
    #mus = tf.Variable(mu_init(shape=[n,d]), name='mus', dtype=tf.float32)
    mus = tf.cast(mu_init(shape=[n,d]), dtype=tf.float32)
    mus = trainable_dist_layer(mus, name = 'mus_%s'%(h_or_c))
    #mus = tf.Variable(initial_value=tf.random.normal([n,d]), name='mus')
    
    # Parameterize the lower-triangular covariance matrix for the Gaussian distribution
    lower_tril_init = tf.initializers.RandomNormal(-(1.0 / n), (1.0 / n))
    rhos = tf.cast(lower_tril_init(shape=[n,tril_components]), dtype=tf.float32)
    rhos = trainable_dist_layer(rhos, name='rhos_%s'%(h_or_c))
    #rhos = tf.Variable(lower_tril_init(shape=([n, tril_components])), name='rhos', dtype=tf.float32)
    #rhos = tf.Variable([n, tril_components], initial_value=tf.initializers.random_normal(-(1.0 / n), (1.0 / n)), name='rhos')

    # The diagonal of the lower-triangular matrix has to be positive, so transform the diagonal with a softplus and then translate it by min_variance.
    scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(min_variance)]))(rhos.return_vars())

    # Make the fully covariant Gaussian distribution
    comp_dist = tfp.distributions.MultivariateNormalTriL(loc=mus.return_vars(), scale_tril=scale_tril)

    # Make the mixture distribution 
    dist = tfp.distributions.MixtureSameFamily(
        components_distribution=comp_dist,
        mixture_distribution=mix_dist,
        name='marginal_dist_%s'%(h_or_c),
        )
    print('MARGINAL trainable params',dist.trainable_variables) 
  
    return dist

class trainable_dist_layer(tf.keras.layers.Layer):
    def __init__(self, init_value, name):
        super(trainable_dist_layer, self).__init__()

        self.trainable_vars = tf.Variable(initial_value=init_value, name = name)

    def return_vars(self):
        return self.trainable_vars
    
    def call(self):
        return self.trainable_vars

