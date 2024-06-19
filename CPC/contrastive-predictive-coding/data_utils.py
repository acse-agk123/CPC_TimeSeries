import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeSeriesHandler(object):
    ''' Provides a convenient interface to manipulate time series data '''

    def __init__(self):
        # Load dataset
        self.X_train, self.X_val, self.X_test = self.load_dataset()

    def load_dataset(self):
        # Load time series data from .npy file
        data = np.load('resources/timeseries_data.npy')

        # Split data into train, validation, and test sets
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        X_train = data[:train_size]
        X_val = data[train_size:train_size + val_size]
        X_test = data[train_size + val_size:]

        return X_train, X_val, X_test

    def process_batch(self, batch, rescale=True):
        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1
        return batch

    def get_batch(self, subset, batch_size, window_size=64, rescale=True):
        # Select a subset
        if subset == 'train':
            X = self.X_train
        elif subset == 'valid':
            X = self.X_val
        elif subset == 'test':
            X = self.X_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, :window_size, :]

        # Process batch
        batch = self.process_batch(batch, rescale)

        return batch.astype('float32')

    def get_n_samples(self, subset):
        if subset == 'train':
            return self.X_train.shape[0]
        elif subset == 'valid':
            return self.X_val.shape[0]
        elif subset == 'test':
            return self.X_test.shape[0]

class TimeSeriesGenerator(tf.keras.utils.Sequence):
    ''' Data generator providing time series data '''

    def __init__(self, batch_size, subset, window_size=64, terms=4, predict_terms=4, num_features=5, rescale=True, verbose=False):
        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.window_size = window_size
        self.terms = terms
        self.predict_terms = predict_terms
        self.num_features = num_features
        self.rescale = rescale
        self.verbose = verbose

        # Initialize time series dataset
        self.handler = TimeSeriesHandler()
        self.n_samples = self.handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        start_time = time.time()
        # Get data
        x = self.handler.get_batch(self.subset, self.batch_size, self.window_size, self.rescale)
        y = self.handler.get_batch(self.subset, self.batch_size, self.window_size, self.rescale)
        if self.verbose:
            print(f"Batch generation took {time.time() - start_time:.2f} seconds")
        x = x.reshape((self.batch_size, self.terms, self.window_size, self.num_features, 1))
        y = y.reshape((self.batch_size, self.predict_terms, self.window_size, self.num_features, 1))
        return (x, y), None  # No labels needed

def create_dataset(generator):
    print("Creating dataset...")
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            (tf.TensorSpec(shape=(generator.batch_size, generator.terms, generator.window_size, generator.num_features, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(generator.batch_size, generator.predict_terms, generator.window_size, generator.num_features, 1), dtype=tf.float32)),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    print("Dataset created.")
    return dataset

def plot_sequences(x, y, labels=None, output_path=None):
    ''' Draws a plot where sequences of numbers can be studied conveniently '''
    sequences = np.concatenate([x, y], axis=1)
    n_batches = sequences.shape[0]
    n_terms = sequences.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.plot(sequences[n_b, n_t, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b, 0])
    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()

if __name__ == "__main__":
    # Test TimeSeriesGenerator
    generator = TimeSeriesGenerator(batch_size=8, subset='train', window_size=64, terms=4, predict_terms=4, num_features=5, rescale=True, verbose=True)
    for (x, y), _ in generator:
        plot_sequences(x, y, output_path=r'resources/batch_sample_timeseries.png')
        break
