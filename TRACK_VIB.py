from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Reshape, TimeDistributed, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softplus
from tensorflow.keras.losses import MSE
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp

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
    
    sense_pos_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_enc')
    
    sense_sal_enc = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_enc')
    
    sal_enc_map = Dense(units=256)
    
    fuse_1_enc = LSTM(units=256, return_sequences=True, return_state=True)
    
    sense_pos_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_1_dec')
    
    sense_sal_dec = LSTM(units=256, return_sequences=True, return_state=True, name='prop_lstm_2_dec')
    
    sal_dec_map = sal_enc_map
    #sal_dec_map = Dense(units=256)
    
    fuse_1_dec = LSTM(units=256, return_sequences=True, return_state=True)
    
    fuse_2 = Dense(units=256)
    
    fc_layer_out = Dense(3)
    To_Position = Lambda(toPosition)
    
    # Encoding
    out_enc_pos, state_h_1, state_c_1 = sense_pos_enc(encoder_position_inputs)
    states_1 = [state_h_1, state_c_1]
    
    out_flat_enc = TimeDistributed(Flatten())(encoder_saliency_inputs)
    out_enc_sal, state_h_2, state_c_2 = sense_sal_enc(out_flat_enc)
    states_2 = [state_h_2, state_c_2]
    
    out_map_sal = TimeDistributed(sal_enc_map)(out_enc_sal)
    mu = out_map_sal[:, :, :128]
    sigma = out_map_sal[:, :, 128:]
    
    encoding_distributions = []
    saliency_samples = []
    
    for m in range(M_WINDOW):
        mu_m = mu[:, m, :]
        sigma_m = sigma[:, m, :]
        m_distribution = tfp.distributions.Normal(mu_m, softplus(sigma_m - BIAS))
        encoding_distributions.append(m_distribution)
        saliency_samples.append(m_distribution.sample())
    
    conc_out_enc = Concatenate(axis=-1)([tf.stack(saliency_samples, axis=1), out_enc_pos])
    
    fuse_out_enc, state_h_fuse, state_c_fuse = fuse_1_enc(conc_out_enc)
    states_fuse = [state_h_fuse, state_c_fuse]
    
    # Decoding
    all_pos_outputs = []
    decoding_distributions = []
    inputs = decoder_position_inputs
    for curr_idx in range(H_WINDOW):
        out_dec_pos, state_h_1, state_c_1 = sense_pos_dec(inputs, initial_state=states_1)
        states_1 = [state_h_1, state_c_1]
    
        selected_timestep_saliency = Lambda(selectImageInModel, arguments={'curr_idx': curr_idx})(decoder_saliency_inputs)
        flatten_timestep_saliency = Reshape((1, NUM_TILES_WIDTH * NUM_TILES_HEIGHT))(selected_timestep_saliency)
        out_dec_sal, state_h_2, state_c_2 = sense_sal_dec(flatten_timestep_saliency, initial_state=states_2)
        states_2 = [state_h_2, state_c_2]
    
        out_map_sal = TimeDistributed(sal_dec_map)(out_dec_sal)
        mu = out_map_sal[:, :, :128]
        sigma = out_map_sal[:, :, 128:]
    
        h_distribution = tfp.distributions.Normal(mu, softplus(sigma - BIAS))
        decoding_distributions.append(h_distribution)
    
        conc_out_dec = Concatenate(axis=-1)([h_distribution.sample(), out_dec_pos])
    
        fuse_out_dec_1, state_h_fuse, state_c_fuse = fuse_1_dec(conc_out_dec, initial_state=states_fuse)
        states_fuse = [state_h_fuse, state_c_fuse]
        fuse_out_dec_2 = TimeDistributed(fuse_2)(fuse_out_dec_1)
    
        outputs_delta = fc_layer_out(fuse_out_dec_2)
    
        decoder_pred = To_Position([inputs, outputs_delta])
    
        all_pos_outputs.append(decoder_pred)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = decoder_pred
    
    # Concatenate all predictions
    decoder_outputs_pos = Lambda(lambda x: K.concatenate(x, axis=1))(all_pos_outputs)

    prior = tfp.distributions.Normal(0.0, 1.0)
    
    def vib_loss(y_true, y_pred):
        class_loss = MSE(y_true, y_pred)
        info_loss_enc = tf.compat.v1.reduce_sum([KL(P, prior) for P in encoding_distributions])
        info_loss_dec = tf.compat.v1.reduce_sum([KL(P, prior) for P in decoding_distributions])
        return class_loss + BETA * info_loss_enc + BETA * info_loss_dec
    
    def KL(P, Q):
        return tf.compat.v1.reduce_sum(tf.compat.v1.reduce_mean(tfp.distributions.kl_divergence(P, Q), 0))
    
    def kl_divergence_enc(y_true, y_pred):
        info_loss = tf.compat.v1.reduce_sum([KL(P, prior) for P in encoding_distributions])
        return info_loss
    
    def kl_divergence_dec(y_true, y_pred):
        info_loss = tf.compat.v1.reduce_sum([KL(P, prior) for P in decoding_distributions])
        return info_loss
    
    # Define and compile model
    model = Model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs], decoder_outputs_pos)
    opt = Adam(LR)
    model.compile(optimizer=opt, loss=vib_loss, metrics=[metric_orth_dist, kl_divergence_enc, kl_divergence_dec])
    return model
