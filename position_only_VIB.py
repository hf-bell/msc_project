from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from tensorflow.keras.activations import softplus
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
import numpy as np
#from Utils import compute_mse
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
    decoder_dense_mot = Dense(2, activation='sigmoid')
    decoder_dense_dir = Dense(2, activation='tanh')
    decoder_params = Dense(2,activation='relu')
    #decoder_df = Dense(2,activation='relu')
    To_Position = Lambda(toPosition)

    # Encoding
    encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_inputs)
    state_h = h_state_mapper(state_h)
    state_c = c_state_mapper(state_c)
    mu_h, sigma_h = state_h[:, :512], state_h[:, 512:]
    mu_c, sigma_c = state_c[:, :512], state_c[:, 512:]
    #n = 1
    #scale_tril_h = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(1e-5)]))(sigma_h)
    #scale_tril_c = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(1e-5)]))(sigma_c)
    #state_h_distribution = tfp.distributions.Normal(tf.cast(mu_h, dtype=tf.float32), softplus(tf.cast(sigma_h, dtype=tf.float32) - 5))
    state_h_distribution = tfp.distributions.MultivariateNormalDiag(mu_h, softplus(sigma_h - 5))
    #state_h_distribution = tfp.distributions.MultivariateNormalTriL(mu_h, scale_tril=scale_tril_h)
    state_h = state_h_distribution.sample()
    state_h_sample = state_h
    #state_c_distribution = tfp.distributions.Normal(tf.cast(mu_c, dtype=tf.float32), softplus(tf.cast(sigma_c, dtype=tf.float32) - 5))
    #state_c_distribution = tfp.distributions.MultivariateNormalTriL(mu_c, scale_tril=scale_tril_c)
    state_c_distribution = tfp.distributions.MultivariateNormalDiag(mu_c, softplus(sigma_c - 5))
    state_c = state_c_distribution.sample()
    state_c_sample = state_c
    #df_init = tf.initializers.RandomNormal(0, 4)
    #outputs_df = df_init(shape=(1,1,2))
    #outputs_df = trainable_dist_layer(outputs_df, name='outputs_df')
    #UQ_samples = 4
    #h_UQ_samples = []
    #c_UQ_samples = []
    #for i in range(UQ_samples):
    #    h_UQ_samples.append(state_h_distribution.sample())
    #    c_UQ_samples.append(state_c_distribution.sample())
     
    states = [state_h, state_c]

    # Decoding
    all_outputs = []
    inputs = decoder_inputs
    outs_dists = []
    for curr_idx in range(H_WINDOW):
        # # Run the decoder on one timestep
        decoder_pred, state_h, state_c = lstm_layer_dec(inputs, initial_state=states)
        outputs_delta = decoder_dense_mot(decoder_pred)
        #f = open("loss.txt", 'a')
        #print("OUTpUTS DELTA CHECK",outputs_delta, file=f)
        #outputs_delta_mu, outputs_delta_sig = outputs_delta[:,:2], outputs_delta[:,2:]
        outputs_delta_dir = decoder_dense_dir(decoder_pred)
        #print("DF",out_df, file=f)
        #print("SIG",out_sig,file=f)
        #outputs_params = decoder_params(decoder_pred)
        #outputs_df, outputs_sig  = outputs_params[:,:,:2], outputs_params[:,:,2:]
        #print("DF MAp",outputs_df, file=f)
        #outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
        outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
        #outputs_pos_dist = tfp.distributions.StudentT(df=outputs_df.return_vars(),
        #        loc=outputs_pos, scale=softplus(outputs_params - 5))
        #outputs_pos_dist = tfp.distributions.MultivariateNormalDiag(
        #        outputs_pos, 
        #        softplus(outputs_params - 5))
        #outs_dists.append(outputs_pos_dist)
        #outputs_pos = outputs_pos_dist.sample()
        # Store the current prediction (we will concantenate all predictions later)
        all_outputs.append(outputs_pos)
        # Reinject the outputs as inputs for the next loop iteration as well as update the states
        inputs = outputs_pos
        states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    # decoder_outputs = all_outputs
    # BETA_h = 1e-5
    # BETA_c = 1e-6
    #f = open("loss.txt", 'w')
    #print(decoder_outputs, file=f)
    marginal_h = marginal_tril_dist('h')
    marginal_c = marginal_tril_dist('c')
    def vib_loss(y_true, y_pred):
        print("loss process")
        class_loss = metric_orth_dist(y_true, y_pred)
        info_loss_h = state_h_distribution.log_prob(state_h_sample)
        info_loss_c = state_c_distribution.log_prob(state_c_sample)
        f = open("loss.txt", 'a')
        #print(y_true, file=f)
        #dec_loss = tf.compat.v1.reduce_sum([outs_dists[i].log_prob(y_true[i]) for i in range(len(outs_dists))])
        #print("Sample",state_c_sample, file=f)
        #print("dist", state_c_distribution, file=f)
        marginal_loss_h = marginal_h.log_prob(tf.cast(state_h_sample, tf.float32))

        #    state_h_sample, tf.float32))
        marginal_loss_c = marginal_c.log_prob(tf.cast(state_c_sample, tf.float32))
        print("MARGINAL LOSS", marginal_loss_c)
        #print("m_c",marginal_loss_c.shape, file = f)
        #print("i_c", info_loss_c.shape, file = f)
        #return class_loss
        return class_loss + BETA_h * (info_loss_h - marginal_loss_h) + BETA_c * (info_loss_c - marginal_loss_c)
        #return BETA_h * (info_loss_h - marginal_loss_h) + BETA_c * (info_loss_c - marginal_loss_c) - dec_loss
    #run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True) 
    # Define and compile model
    model = MarginalGradModel([encoder_inputs, decoder_inputs], decoder_outputs)
    opt = Adam(5e-4)
    print("before compile")
    model.compile(optimizer=opt, loss=vib_loss, metrics=[metric_orth_dist])
        #run_eagerly=True)
    #f = open("loss.txt",'a')
    #print(model, file=f)
    #mag_rate = tf.compat.v1.reduce_sum([outs_mag_dists[i].log_prob(outputs_mag[i]) for i in range(len(outs_mag_dists))]) - marginal_h.log_prob(state_h_distribution.mean())
    h_rate = state_h_distribution.log_prob(state_h_distribution.mean()) - marginal_h.log_prob(state_h_distribution.mean())
    c_rate = state_c_distribution.log_prob(state_c_distribution.mean()) - marginal_c.log_prob(state_c_distribution.mean()) 
    dist_model = Model([encoder_inputs, decoder_inputs], [h_rate,c_rate])
    #dist_model = Model([encoder_inputs, decoder_inputs], [mag_rate])
    acts_model = Model([encoder_inputs, decoder_inputs], [state_h_sample, state_c_sample])
    check_model = Model([encoder_inputs, decoder_inputs], [outputs_delta])
    #data_UQ_model = Model([encoder_inputs, decoder_inputs], [h_UQ_samples + c_UQ_samples])
    #Model([encoder_inputs, decoder_inputs], [h_UQ_samples, c_UQ_samples])
    #dist_model = Model([encoder_inputs, decoder_inputs], [state_h_distribution.loc,state_h_distribution.stddev(),state_c_distribution.loc,state_c_distribution.stddev()]) 
    #opt = Adam(5e-4)
    #tf.compat.v1.enable_eager_execution()
    f = open("loss.txt", 'a')
    output = model.output
    #gradients = K.gradients(output, marginal_weights)
    #print("GRADIENTS", gradients, file=f)
    return {'model':model, 'dist_model':dist_model, 'acts_model':acts_model, 'check_model':check_model}


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

#def vib_loss(y_true, y_pred):
        #class_loss = metric_orth_dist(y_true, y_pred)
        #info_loss_h = state_h_distribution.log_prob(state_h_sample)
        #info_loss_c = state_c_distribution.log_prob(state_c_sample)
        #f = open("loss.txt", 'a')
        #print(y_true, file=f)
        #dec_loss = tf.compat.v1.reduce_sum([outs_dists[i].log_prob(y_true[i]) for i in range(len(outs_dists))])
        #print("Sample",state_c_sample, file=f)
        #print("dist", state_c_distribution, file=f)
        #marginal_loss_h = marginal_h.log_prob(tf.cast(state_h_sample, tf.float32))

        #    state_h_sample, tf.float32))
        #marginal_loss_c = marginal_c.log_prob(tf.cast(state_c_sample, tf.float32))

        #print("m_c",marginal_loss_c.shape, file = f)
        #print("i_c", info_loss_c.shape, file = f)
        #return class_loss + BETA_h * (info_loss_h - marginal_loss_h) + BETA_c * (info_loss_c - marginal_loss_c)

class MarginalGradModel(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #x, y = data
        x, y = data
        x1, x2 = x[0], x[1]
        with tf.GradientTape() as tape:
            y_pred = self([x1,x2], training=True)  # Forward pass
            print("YYYYYYYYYYYYYYYYY",y_pred)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            print("LOSSES",self.losses)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #loss = vib_loss(y, y_pred) 
        print("BEFORE TRAINABLE")
        print("LOSS",loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        #marginal_h_vars = marginal_h.trainable_variables
        #marginal_c_vars = marginal_c.trainable_variables
        sess = K.get_session()
        print("BEFORE VARS LOO")
        for v in tf.compat.v1.trainable_variables():
            if "mu" in v.name:
                if "c" in v.name:
                    mus_c = v
                    print(v)
                else:
                    mus_h = v
                    print(v)
            elif "rho" in v.name:
                if "c" in v.name:
                    rhos_c = v
                    print(v)
                else:
                    rhos_h = v
                    print(v)
            elif "mix_logits" in v.name:
                if "c" in v.name:
                    mix_logits_c = v
                    print(v)
                else:
                    mix_logits_h = v
                    print(v)
        #print(marginal_h_vars,"---------------", file=f)
        gradients = tape.gradient(loss, trainable_vars)
        print("before grads mus")
        mus_h_gradients = tape.gradient(loss, mus_h)
        rhos_h_gradients = tape.gradient(loss, rhos_h)
        mix_logits_h_gradients = tape.gradient(loss, mix_logits_h)

        mus_c_gradients = tape.gradient(loss, mus_c)
        rhos_c_gradients = tape.gradient(loss, rhos_c)
        mix_logits_c_gradients = tape.gradient(loss, mix_logits_c)
        
        #marginal_c_gradients = tape.gradient(loss, marginal_c_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.optimizer.apply_gradients(zip(mus_h_gradients, mus_h))
        self.optimizer.apply_gradients(zip(rhos_h_gradients, rhos_h))
        self.optimizer.apply_gradients(zip(mix_logits_h_gradients, mix_logits_h))
        self.optimizer.apply_gradients(zip(mus_c_gradients, mus_c))
        self.optimizer.apply_gradients(zip(rhos_c_gradients, rhos_c))
        self.optimizer.apply_gradients(zip(mix_logits_c_gradients, mix_logits_c))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
