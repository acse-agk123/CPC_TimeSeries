import time
from data_utils import TimeSeriesGenerator, create_dataset
from os.path import join
import keras
from keras import backend as K
from keras.layers import Lambda
import tensorflow as tf

def network_encoder(input_shape, code_size):
    ''' Define the network mapping time series to embeddings '''
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)
    return keras.models.Model(inputs, x, name='encoder')

def network_autoregressive(x):
    ''' Define the network that integrates information along the sequence '''
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    return x

def network_prediction(context, code_size, predict_terms):
    ''' Define the network mapping context to multiple embeddings '''
    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name=f'z_t_{i}')(context))
    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), output_shape=(1, code_size))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: tf.stack(x, axis=1), output_shape=(predict_terms, code_size))(outputs)
    return output

class CPCLayer(keras.layers.Layer):
    ''' Computes dot product between true and predicted embedding vectors '''
    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = tf.reduce_mean(y_encoded * preds, axis=-1)
        dot_product = tf.reduce_mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
        # Keras loss functions take probabilities
        dot_product_probs = tf.sigmoid(dot_product)
        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

def network_cpc(input_shape, terms, predict_terms, code_size, learning_rate):
    ''' Define the CPC network combining encoder and autoregressive model '''
    print("Defining encoder model...")
    encoder_model = network_encoder(input_shape, code_size)
    encoder_model.summary()

    # Define rest of model
    print("Defining the rest of the model...")
    x_input = keras.layers.Input((terms, input_shape[0], input_shape[1], input_shape[2]))
    x_encoded = Lambda(lambda x: tf.map_fn(lambda t: encoder_model(t), x, fn_output_signature=tf.float32), output_shape=(terms, code_size))(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, input_shape[0], input_shape[1], input_shape[2]))
    y_encoded = Lambda(lambda y: tf.map_fn(lambda t: encoder_model(t), y, fn_output_signature=tf.float32), output_shape=(predict_terms, code_size))(y_input)

    # Loss
    print("Defining CPC layer for loss computation...")
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    print("Creating CPC model...")
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    print("Compiling the model...")
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    cpc_model.summary()

    return cpc_model

class TimingCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"Epoch {epoch + 1} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds")

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_duration = time.time() - self.batch_start_time
        print(f"Batch {batch + 1} took {batch_duration:.2f} seconds")

def train_model(epochs, batch_size, output_dir, code_size, learning_rate=1e-4, terms=4, predict_terms=4, window_size=64, num_features=5):
    start_time = time.time()
    print("Preparing data...")
    # Prepare data
    train_data_generator = TimeSeriesGenerator(batch_size=batch_size, subset='train', window_size=window_size,
                                               terms=terms, predict_terms=predict_terms, num_features=num_features, rescale=True, verbose=True)
    validation_data_generator = TimeSeriesGenerator(batch_size=batch_size, subset='valid', window_size=window_size,
                                                    terms=terms, predict_terms=predict_terms, num_features=num_features, rescale=True, verbose=True)

    train_data = create_dataset(train_data_generator)
    validation_data = create_dataset(validation_data_generator)

    print(f"Data preparation took {time.time() - start_time:.2f} seconds")

    # Prepares the model
    print("Preparing the model...")
    model = network_cpc(input_shape=(window_size, num_features, 1), terms=terms, predict_terms=predict_terms,
                        code_size=code_size, learning_rate=learning_rate)

    # Callbacks
    print("Setting up callbacks...")
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4),
        TimingCallback()
    ]

    print("Starting model training...")
    train_start_time = time.time()

    # Trains the model
    print("Calling model.fit...")
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        verbose=2,  # Set verbose to 2 for one line per epoch
        callbacks=callbacks
    )

    print(f"Model training took {time.time() - train_start_time:.2f} seconds")

    # Saves the model
    print("Saving the model...")
    model.save(join(output_dir, 'cpc.h5'))

    # Saves the encoder alone
    print("Saving the encoder model...")
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder.h5'))

if __name__ == "__main__":
    print("Starting training script...")
    train_model(
        epochs=10,
        batch_size=32,
        output_dir='models/timeseries',
        code_size=128,
        learning_rate=1e-3,
        terms=4,
        predict_terms=4,
        window_size=64,
        num_features=5
    )
    print("Training script completed.")
