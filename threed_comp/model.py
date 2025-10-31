# VAE model implementation from https://www.kaggle.com/code/tasadanluca/3d-object-morphing#Building-the-VAE-model
import tensorflow as tf
from tensorflow.keras import layers
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

L2_WEIGHT = 1e-5
GAMMA = 0.97
def weighted_bce_loss(y_true, y_pred, gamma=GAMMA):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    gamma_t = tf.cast(gamma, tf.float32)
    
    y_pred = (y_pred - 0.1) / 0.9
    y_true = (y_true + 1.0) / 3.0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = -gamma * y_true * K.log(y_pred) - (1 - gamma) * (1 - y_true) * K.log(1 - y_pred)
    return loss

class Sampling(layers.Layer):
    """
    This class uses the z_mean and z_log_var in order to sample z from the distribution
    """
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape = (batch,dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon

def scaled_sigmoid_activation(x):
    sigmoid = tf.math.sigmoid(x)
    return (sigmoid * 0.9) +0.1

# In this cell I will introduce some hyperparameters and methods that mostly prevent the model from learning that
# an empty space produces the best loss, as I had this problem in my earlier versions.
# Thus I will introduce L2 regularization and a new weighted loss functions


L2_WEIGHT = 1e-5
GAMMA = 0.97
def weighted_bce_loss(y_true, y_pred, gamma):
    """
    Implements the specialized BCE loss function.
    L = -γ * t * log(o) - (1-γ) * (1-t) * log(1-o)
    """
    # Making sure all are of type float32
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    gamma_t = tf.cast(gamma, tf.float32)
    
    y_pred = (y_pred - 0.1) / 0.9
    y_true = (y_true + 1.0) / 3.0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = -gamma * y_true * K.log(y_pred) - (1 - gamma) * (1 - y_true) * K.log(1 - y_pred)
    return loss



def build_encoder(input_dim, latent_dim, l2_weight):
    encoder_inputs = layers.Input(shape = input_dim)
    # Downsampling by using strides = 2
    x = layers.Conv3D(32, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(l2_weight))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(64, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    
    # Residual Block
    residual = x
    x = layers.Conv3D(128, 3, activation='elu', strides=2, padding='same',kernel_regularizer=l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, 3, activation='elu', padding='same',kernel_regularizer=l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut connection for the resnet architecture
    residual_downsampled = layers.Conv3D(128, 1, strides=2, padding='same')(residual)
    x = layers.add([x,residual_downsampled])
    
    # Flatten and map to the latent space
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation = 'elu')(x)

    # Obtaining the mean and log variance
    z_mean = layers.Dense(latent_dim, name = 'z_mean', kernel_regularizer = l2(l2_weight))(x)
    z_log_var = layers.Dense(latent_dim, name = 'z_log_var', kernel_regularizer = l2(l2_weight))(x)

    z = Sampling()([z_mean,z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean,z_log_var,z], name = 'encoder')
    return encoder

def build_decoder(latent_dim, reshape_dims, l2_weight):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='elu')(latent_inputs)

    #Upsample from the latent vector to the small grid necessary for the decoder
    x = layers.Dense(reshape_dims[0] * reshape_dims[1] * reshape_dims[2] * 128, activation="elu")(x)
    x = layers.Reshape(reshape_dims)(x)

    
     # The transposed residual block
    residual = x
    x = layers.Conv3DTranspose(128, 3, activation='elu', padding='same',
                               kernel_regularizer=l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3DTranspose(128, 3, activation='elu', strides=2, padding='same',
                               kernel_regularizer=l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    residual = layers.Conv3DTranspose(128, 1, strides=2, padding='same')(residual)
    x = layers.add([x, residual])

    # Continue the upsampling
    x = layers.Conv3DTranspose(64, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3DTranspose(32, 3, activation = 'elu', strides = 2, padding = 'same', kernel_regularizer = l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)

    # This is the final layer that reconstructs the voxel grid
    decoder_outputs = layers.Conv3DTranspose(1, 3, activation=scaled_sigmoid_activation, padding="same", kernel_regularizer = l2(l2_weight))(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name = 'decoder')
    return decoder


class VAE(keras.Model):
    """
    Variational Autoencoder model for 3D voxel data.
    encoder = build_encoder(input_dim, latent_dim, L2_WEIGHT)
    decoder = build_decoder(latent_dim, reshape_dim, L2_WEIGHT)
    vae = VAE(encoder,decoder, beta = 10)

    """
    def __init__(self, encoder, decoder, beta, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name = 'kl_loss')

    def call(self, inputs):
        """Defines the forward pass of the model."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, targets = data
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    weighted_bce_loss(targets, reconstruction, gamma = GAMMA),
                    axis=(1, 2,3),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss + tf.reduce_sum(self.losses)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def test_step(self, data):
        """Defines the logic for one evaluation step."""
        x, targets = data
        
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                    weighted_bce_loss(targets, reconstruction, gamma = GAMMA),
                axis=(1, 2, 3),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        total_loss = reconstruction_loss + self.beta * kl_loss + tf.reduce_sum(self.losses)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}

