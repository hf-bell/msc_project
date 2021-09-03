from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Reshape, TimeDistributed, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import MSE
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import numpy as np

from models_utils import metric_orth_dist, selectImageInModel, toPosition

BIAS = 5
BETA = 1e-6


# ----------------------------- TRAIN ----------------------------
def create_TRACK_VIB_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, LR):
    # Defining model structure
    encoder_position_inputs = Input(shape=(M_WINDOW, 3))
    encoder_saliency_inputs = Input(shape=(M_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))
    decoder_position_inputs = Input(shape=(1, 3))
    decoder_saliency_inputs = Input(shape=(H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH, 1))

    # inertia & content encoder LSTMs
    inertia_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_enc')
    content_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_enc')

    # dense layer to extract mean & variance from
    sal_enc_map = Dense(units=256)
    fuse_1_enc = LSTM(units=256, return_sequences=True, return_state=True)

    # inertia & content decoder LSTMs
    inertia_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_dec')
    content_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_dec')

    #sal_dec_map = sal_enc_map
    sal_dec_map = Dense(units=256)

    fuse_1_dec = LSTM(units=256, return_sequences=True, return_state=True)

    fuse_2 = Dense(units=256)

    fc_layer_out = Dense(3)
    To_Position = Lambda(toPosition)


################################ ENCODING  ##########################################
    # Encode head position history w/ inertia LSTM
    out_enc_pos, state_h_1, state_c_1 = inertia_enc(encoder_position_inputs)
    inertia_states = [state_h_1, state_c_1]

    # Encode visual content history w/ content LSTM
    out_flat_enc = TimeDistributed(Flatten())(encoder_saliency_inputs) # TimeDistributed splits video into temporal slices
                                                                    # - allows application of content LSTM to video content 
    out_enc_sal, state_h_2, state_c_2 = content_enc(out_flat_enc)
    content_states = [state_h_2, state_c_2]

    # At each timestep, pass content RNN outputs
    # through FC layer to generate mean & variance values
    enc_map_sal = TimeDistributed(sal_enc_map)(out_enc_sal)
    mu = enc_map_sal[:, :, :128]
    sigma = enc_map_sal[:, :, 128:]

    encoding_distributions = []
    saliency_samples = []

    # Take reparameterised samples from latent variables
    # for each encoder timestep
    for m in range(M_WINDOW):
        mu_m = mu[:, m, :]
        sigma_m = sigma[:, m, :]
        m_distribution = tfp.distributions.Normal(mu_m, softplus(sigma_m - BIAS))
        # Store VIB output distribution for use in loss
        encoding_distributions.append(m_distribution)
        #saliency_samples.append(m_distribution.sample(sample_shape=[3]))
        saliency_samples.append(m_distribution.sample())

    # concatenate samples & inertia RNN outputs
    conc_out_enc = Concatenate(axis=-1)([tf.stack(saliency_samples, axis=1), out_enc_pos])

    # pass concatenated outputs to Fusion RNN
    fuse_out_enc, state_h_fuse, state_c_fuse = fuse_1_enc(conc_out_enc)
    states_fuse = [state_h_fuse, state_c_fuse]

################################ DECODING  ##########################################
    all_pos_outputs = []
    decoding_distributions = []
    output_encodings = []
    inputs = decoder_position_inputs
    for curr_idx in range(H_WINDOW):
        out_dec_pos, state_h_1, state_c_1 = inertia_dec(inputs, initial_state=inertia_states)
        inertia_states = [state_h_1, state_c_1]
        selected_timestep_saliency = Lambda(selectImageInModel, arguments={'curr_idx': curr_idx})(decoder_saliency_inputs)
        flatten_timestep_saliency = Reshape((1, NUM_TILES_WIDTH * NUM_TILES_HEIGHT))(selected_timestep_saliency)
        out_dec_sal, state_h_2, state_c_2 = content_dec(flatten_timestep_saliency, initial_state= content_states)
        content_states = [state_h_2, state_c_2]

        #dec_logits = TimeDistributed(sal_dec_map)(out_dec_sal)
        out_map_sal = TimeDistributed(sal_dec_map)(out_dec_sal)


        mu = out_map_sal[:, :, :128]
        sigma = out_map_sal[:, :, 128:]
        h_distribution = tfp.distributions.Normal(mu, softplus(sigma - BIAS))
        #h_distribution = tfp.distributions.Categorical(logits=dec_logits)

        # Store VIB output distribution for use in loss
        decoding_distributions.append(h_distribution)
        #conc_out_dec = Concatenate(axis=-1)([tf.cast(h_distribution.sample(sample_shape=[256]), tf.float32), out_dec_pos])
        conc_out_dec = Concatenate(axis=-1)([tf.cast(h_distribution.sample(), tf.float32), out_dec_pos])

        fuse_out_dec_1, state_h_fuse, state_c_fuse = fuse_1_dec(conc_out_dec, initial_state=states_fuse)
        states_fuse = [state_h_fuse, state_c_fuse]
        fuse_out_dec_2 = TimeDistributed(fuse_2)(fuse_out_dec_1)

        outputs_delta = fc_layer_out(fuse_out_dec_2)
        output_encodings.append(outputs_delta)

        decoder_pred = To_Position([inputs, outputs_delta])

        all_pos_outputs.append(decoder_pred)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = decoder_pred

    # Concatenate all predictions
    decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(all_pos_outputs)

    #prior = tfp.distributions.Normal(0.0, 1.0) # for prior
    marginal = marginal_tril_dist()

    def vib_loss(y_true, y_pred):
        class_loss = MSE(y_true, y_pred)
        info_loss_enc = tf.reduce_sum([encoding_distributions[n].log_prob(saliency_samples[n]) for n in range(0, len(encoding_distributions))])
        #info_loss_dec = [encoding_distributions[-1].log_prob(output) for output in output_encodings]
        marginal_loss = tf.reduce_sum([marginal.log_prob(tf.cast(z_samp, tf.float32)) for z_samp in saliency_samples])
        return class_loss + BETA * (info_loss_enc - marginal_loss)

    # Define and compile model
    model = Model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, \
                   decoder_saliency_inputs], decoder_outputs_pos)
    opt = Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=vib_loss, metrics=[metric_orth_dist, vib_loss])
    return model

# Define learnable marginal distribution for UQ
def marginal_tril_dist():
    n = 200
    d = 128
    min_variance = 1e-5
    # Compute the number of parameters needed for the lower triangular covariance matrix
    tril_components = (d  * (d + 1)) // 2

    # Parameterize the categorical distribution for the mixture
    #mix_logits = tf.Variable('mix_logits', [n]) # get or create params to fill marginal dist.
    init_vals = np.random.rand(n)
    init_probs = [float(i)/sum(init_vals) for i in init_vals]
    mix_logits = tf.cast(np.log(init_probs), dtype=tf.float32)
    mix_logits = trainable_dist_layer(mix_logits, name='mix_logits')
    #mix_logits = tf.Variable(initial_value=np.log(init_probs), name='mix_logits', dtype=tf.float32)

    mix_dist = tfp.distributions.Categorical(logits=mix_logits.return_vars())

    # Parameterize the means of Gaussian distribution
    mu_init = tf.initializers.RandomNormal()
    #mus = tf.Variable(mu_init(shape=[n,d]), name='mus', dtype=tf.float32)
    mus = tf.cast(mu_init(shape=[n,d]), dtype=tf.float32)
    mus = trainable_dist_layer(mus, name = 'mus')
    #mus = tf.Variable(initial_value=tf.random.normal([n,d]), name='mus')

    # Parameterize the lower-triangular covariance matrix for the Gaussian distribution
    lower_tril_init = tf.initializers.RandomNormal(-(1.0 / n), (1.0 / n))
    rhos = tf.cast(lower_tril_init(shape=[n,tril_components]), dtype=tf.float32)
    rhos = trainable_dist_layer(rhos, name='rhos')
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
        name='marginal_dist',
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
