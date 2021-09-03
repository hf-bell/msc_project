import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Lambda, Input, Reshape, Convolution2D, TimeDistributed, Concatenate, Flatten, ConvLSTM2D, MaxPooling2D
from tensorflow.keras import optimizers, losses
import tensorflow as tf
from tensorflow.keras import backend as K
from models_utils import metric_orth_dist

class Pos_Only_Class:

    def __init__(self, h_window, weights=None):
        self.model = self.create_pos_only_model(h_window)
        self.h_window = h_window
        if weights is not None:
            self.model.load_weights(weights)

    def get_model(self):
        return self.model

    def get_model_scheduled_sampling(self):
        return self.create_pos_only_scheduled_sampling_model(self.h_window)

    def predict(self, pos_inputs, visual_inputs=None):
        pred = self.model.predict([np.array([pos_inputs[:-1]]), np.array([pos_inputs[-1:]])])
        norm_factor = np.sqrt(pred[0, :, 0]*pred[0, :, 0] + pred[0, :, 1]*pred[0, :, 1] + pred[0, :, 2]*pred[0, :, 2])
        data = {'x': pred[0, :, 0]/norm_factor,
                'y': pred[0, :, 1]/norm_factor,
                'z': pred[0, :, 2]/norm_factor
                }
        return pd.DataFrame(data)

    # This way we ensure that the network learns to predict the delta angle
    def toPosition(self, values):
        orientation = values[0]
        delta = values[1]
        return orientation + delta

    def loss_function(self, x_true, x_pred):
        xent_loss = losses.mse(x_true, x_pred)
        unitary_loss = K.square((K.sqrt(K.sum(K.square(x_pred), axis=-1))) - 1.0)
        loss = xent_loss + unitary_loss
        return loss

    def create_pos_only_model(self, h_window):
        # Defining model structure
        encoder_inputs = Input(shape=(None, 3))
        decoder_inputs = Input(shape=(1, 3))

        sense_pos_1 = TimeDistributed(Dense(256))
        sense_pos_2 = TimeDistributed(Dense(256))
        sense_pos_3 = TimeDistributed(Dense(256))
        lstm_layer_enc = LSTM(1024, return_sequences=True, return_state=True)
        lstm_layer_dec = LSTM(1024, return_sequences=True, return_state=True)
        decoder_dense_1 = Dense(256)
        decoder_dense_2 = Dense(256)
        decoder_dense_3 = Dense(3)
        To_Position = Lambda(self.toPosition)

        # Encoding
        encoder_outputs = sense_pos_1(encoder_inputs)
        encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_outputs)
        states = [state_h, state_c]

        # Decoding
        all_outputs = []
        inputs = decoder_inputs
        for curr_idx in range(h_window):
            # # Run the decoder on one timestep
            inputs_1 = sense_pos_1(inputs)
            inputs_2 = sense_pos_2(inputs_1)
            inputs_3 = sense_pos_3(inputs_2)
            decoder_pred, state_h, state_c = lstm_layer_dec(inputs_3, initial_state=states)
            outputs_delta = decoder_dense_1(decoder_pred)
            outputs_delta = decoder_dense_2(outputs_delta)
            outputs_delta = decoder_dense_3(outputs_delta)
            outputs_pos = To_Position([inputs, outputs_delta])
            # Store the current prediction (we will concantenate all predictions later)
            all_outputs.append(outputs_pos)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            inputs = outputs_pos
            states = [state_h, state_c]
        if h_window == 1:
            decoder_outputs = outputs_pos
        else:
            # Concatenate all predictions
            decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model_optimizer = optimizers.Adam(lr=0.0005)
        model.compile(optimizer=model_optimizer, loss=self.loss_function, metrics=[metric_orth_dist])
        return model

    def scheduledSampling(self, values):
        return tf.where(values[0], values[1], values[2])

    def selectTimeStepInSample(self, input_to_selector, curr_idx):
        selected_tstep = input_to_selector[:, curr_idx:curr_idx+1]
        return selected_tstep

    def create_pos_only_scheduled_sampling_model(self, h_window):
        # Defining model structure
        encoder_inputs = Input(shape=(None, 3))
        decoder_inputs = Input(shape=(1, 3))
        decoder_groundtruth = Input(shape=(h_window, 3))
        decoder_scheduler_sampler = Input(shape=(h_window, 3), dtype=bool)

        sense_pos_1 = TimeDistributed(Dense(256))
        sense_pos_2 = TimeDistributed(Dense(256))
        sense_pos_3 = TimeDistributed(Dense(256))
        lstm_layer_enc = LSTM(1024, return_sequences=True, return_state=True)
        lstm_layer_dec = LSTM(1024, return_sequences=True, return_state=True)
        decoder_dense_1 = Dense(256)
        decoder_dense_2 = Dense(256)
        decoder_dense_3 = Dense(3)
        To_Position = Lambda(self.toPosition)
        Scheduled_Sampling = Lambda(self.scheduledSampling)

        # Encoding
        encoder_outputs = sense_pos_1(encoder_inputs)
        encoder_outputs, state_h, state_c = lstm_layer_enc(encoder_outputs)
        states = [state_h, state_c]

        # Decoding
        all_outputs = []
        inputs = decoder_inputs
        for curr_idx in range(h_window):
            # # Run the decoder on one timestep
            inputs_1 = sense_pos_1(inputs)
            inputs_2 = sense_pos_2(inputs_1)
            inputs_3 = sense_pos_3(inputs_2)
            decoder_pred, state_h, state_c = lstm_layer_dec(inputs_3, initial_state=states)
            outputs_delta = decoder_dense_1(decoder_pred)
            outputs_delta = decoder_dense_2(outputs_delta)
            outputs_delta = decoder_dense_3(outputs_delta)
            outputs_pos = To_Position([inputs, outputs_delta])
            # Store the current prediction (we will concantenate all predictions later)
            all_outputs.append(outputs_pos)
            # Reinject the outputs as inputs for the next loop iteration as well as update the states
            # But performing scheduled sampling
            tstep_scheduler = Lambda(self.selectTimeStepInSample, arguments={'curr_idx': curr_idx})(decoder_scheduler_sampler)
            tstep_gtruth = Lambda(self.selectTimeStepInSample, arguments={'curr_idx': curr_idx})(decoder_groundtruth)
            inputs = Scheduled_Sampling([tstep_scheduler, tstep_gtruth, outputs_pos])
            states = [state_h, state_c]
        if h_window == 1:
            decoder_outputs = outputs_pos
        else:
            # Concatenate all predictions
            decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model
        model = Model([encoder_inputs, decoder_inputs, decoder_groundtruth, decoder_scheduler_sampler], decoder_outputs)
        model_optimizer = optimizers.Adam(lr=0.0005)
        model.compile(optimizer=model_optimizer, loss=self.loss_function, metrics=[metric_orth_dist])
        return model

    def format_input(self, dataset_reader, pos_inputs):
        coords_3d = dataset_reader.transform_to_3d_coordinates(pos_inputs)
        return coords_3d[['x', 'y', 'z']].values

    def format_output(self, dataset_reader, pos_outputs):
        return dataset_reader.transform_3d_coordinates_to_original_format(pos_outputs)

    def format_dataset(self, dataset_reader, dataset):
        formatted_dataset = {}
        for video_id in dataset.keys():
            formatted_dataset[video_id] = {}
            for user_id in dataset[video_id].keys():
                coords_3d = dataset_reader.transform_to_3d_coordinates(dataset[video_id][user_id])
                formatted_dataset[video_id][user_id] = coords_3d[['x', 'y', 'z']].values
        return formatted_dataset

if __name__ == "__main__":
    batch_size = 9
    m_window = 3
    h_window = 2
    pos_only = Pos_Only_Class(h_window)
    # model = pos_only.get_model()
    # model.fit([np.random.rand(batch_size, h_window-1, 3), np.random.rand(batch_size, 1, 3)], np.random.rand(batch_size, h_window, 3))
    # pred = pos_only.predict(np.random.rand(m_window, 3))

    model = pos_only.get_model_scheduled_sampling()
    model.fit([np.random.rand(batch_size, m_window - 1, 3), np.random.rand(batch_size, 1, 3),
               np.random.rand(batch_size, h_window, 3), np.random.rand(batch_size, h_window, 3) > 0.5],
              np.random.rand(batch_size, h_window, 3))

    pred = model.predict([np.random.rand(batch_size, m_window-1, 3), np.random.rand(batch_size, 1, 3), np.random.rand(batch_size, h_window, 3), np.random.rand(batch_size, h_window, 3) > 0.5])

    print(pred)
