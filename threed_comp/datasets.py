import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class VoxelizedDataset(tf.keras.utils.Sequence):
    def __init__(
        self, filenames, batch_size, augment=False, max_translation=5, **kwargs
    ):
        super().__init__(**kwargs)
        self.filenames = filenames
        self.batch_size = batch_size
        self.augment = augment
        self.max_translation = max_translation

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.filenames) / self.batch_size))

    def apply_augmentations(self, voxel_grid):
        shifts = np.random.randint(
            -self.max_translation, self.max_translation + 1, size=3
        )
        augmented_grid = np.roll(voxel_grid, shifts, axis=(0, 1, 2))

        if np.random.rand() > 0.5:
            augmented_grid = np.flip(augmented_grid, axis=2)

        return augmented_grid

    def __getitem__(self, index):
        batch_filenames = self.filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        original_batch = np.array([np.load(f) for f in batch_filenames])

        if self.augment:
            augmented_batch = np.array(
                [self.apply_augmentations(grid) for grid in original_batch]
            )

            current_batch = np.concatenate([original_batch, augmented_batch], axis=0)

            target_batch = np.concatenate([original_batch, original_batch], axis=0)
        else:
            current_batch = original_batch
            target_batch = original_batch

        modified_batch = np.where(target_batch == 1, 2.0, -1.0)

        current_batch = np.expand_dims(current_batch, axis=-1)
        modified_batch = np.expand_dims(modified_batch, axis=-1)

        return current_batch, modified_batch


class Sampling(layers.Layer):
    """
    This class uses the z_mean and z_log_var in order to sample z from the distribution
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
