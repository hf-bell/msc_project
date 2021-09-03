from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


# Load data -- graph of a [cardioid](https://en.wikipedia.org/wiki/Cardioid).
n = 2000
t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
r = 2 * (1 - tf.cos(t))
x = r * tf.sin(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
y = r * tf.cos(t) + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
BIAS = 5
BETA = 1e-6

# Model the distribution of y given x with a Mixture Density Network.
event_shape = [1]
num_components = 5

enc_dense = tfkl.Dense(12, activation='relu')(x)
mu = enc_dense[:6]
sigma = enc_dense[6:]
dist = tfp.distributions.Normal(mu, softplus(sigma - BIAS))
out = tfkl.Dense(12, activation='relu')(dist.sample())


marginal = marginal_tril_dist()

def vib_loss(y_true, y_pred):
    class_loss = MSE(y_true, y_pred)
    info_loss_enc = tf.compat.v1.reduce_sum([KL(P, marginal) for P in encoding_distributions])
    info_loss_dec = tf.compat.v1.reduce_sum([KL(P, marginal) for P in decoding_distributions])
    return class_loss + BETA * info_loss_enc + BETA * info_loss_dec

def KL(P, Q):
    return tf.compat.v1.reduce_sum(tf.compat.v1.reduce_mean(tfp.distributions.kl_divergence(P, Q), 0))

def kl_divergence_enc(y_true, y_pred):
    info_loss = tf.compat.v1.reduce_sum([KL(P, marginal) for P in encoding_distributions])
    return info_loss

def kl_divergence_dec(y_true, y_pred):
    info_loss = tf.compat.v1.reduce_sum([KL(P, marginal) for P in decoding_distributions])
    return info_loss

# Define and compile model
model = Model([encoder_position_inputs, encoder_saliency_inputs, decoder_position_inputs, decoder_saliency_inputs], decoder_outputs_pos)
opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss=vib_loss, metrics=[metric_orth_dist, kl_divergence_enc, kl_divergence_dec])


model.fit(x, y,
          batch_size=batch_size,
          epochs=20,
          steps_per_epoch=n // batch_size)

model.save('categorical_test.h5')

def marginal_tril_dist():
    n = 10
    d = 3
    min_variance = 1e-5
    # Compute the number of parameters needed for the lower triangular covariance matrix
    tril_components = (d  * (d + 1)) // 2

    init_logits = np.random.rand(,200)
    init_logits = [float(i)/sum(init_logits) for i in init_logits]
    # Parameterize the categorical distribution for the mixture
    mix_logits = tf.Variable('mix_logits', [n]) # get or create params to fill marginal dist.
    print(init_logits)
    
    mix_dist = tfp.distributions.Categorical(logits=init_logits, 200)
    # Parameterize the means of Gaussian distribution
    mus = tf.Variable('mus', [n, d], initializer=tf.initializers.random_normal())
    # Parameterize the lower-triangular covariance matrix for the Gaussian distribution
    rhos = tf.Variable('rhos', [n, tril_components], initializer=tf.initializers.random_normal(-(1.0 / n), (1.0 / n)))
    # The diagonal of the lower-triangular matrix has to be positive, so transform the diagonal with a softplus and then translate it by min_variance.
    scale_tril = tfp.bijectors.FillScaleTriL(diag_bijector=tf.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(min_variance)]))(rhos)


    # Make the fully covariant Gaussian distribution
    comp_dist = tfp.distributions.MultivariateNormalTriL(loc=mus, scale_tril=scale_tril)

    # Make the mixture distribution 
    dist = tfp.distributions.MixtureSameFamily(
        components_distribution=comp_dist,
        mixture_distribution=mix_dist,
        )
  
    return dist
