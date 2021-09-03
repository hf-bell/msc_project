# adapted from https://github.com/bsaund/normalizing_flows/blob/master/normalizing_flows.py
# https://www.bradsaund.com/post/normalizing_flows/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

settings = {
    'num_bijectors': 4,
    train
    }

class Flow(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Flow, self).__init__(**kwargs)
        flow = None

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    @tf.function
    def train_step(self, X, optimizer):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    

class MAF(Flow):
    def __init__(self, layer_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)

        self.num_masked = num_masked

        self.bijector_fns = []

        bijectors = []
        for i in range(settings['num_bijectors']):
            self.bijector_fns.append(tfb.AutoregressiveNetwork(params=2, hidden_units=[512, 512]))
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=self.bijector_fns[-1]
                )
            )

            # if i%2 == 0:
            #     bijectors.append(tfb.BatchNormalization())

            #bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0,0.0]),
            bijector=bijector,
            event_shape=layer_dim)


def train(weights_file, model_name, model, flow_model, partition, optimizer):
    if isinstance(model, dict):
        pred_model = model['model']
        layer_model = model['dist_model']
    model.load_weights(weights_file)
    traces_count = 0
    for ID in partition['train']:
        traces_count += 1

        user = ID['user']
        video = ID['video']
        x_i = ID['time-stamp']

        if model_name != 'pos_only':
            
