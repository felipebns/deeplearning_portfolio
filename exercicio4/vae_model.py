import tensorflow as tf
from tensorflow.keras import Model

class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data, training=True)
            eps = tf.random.normal(tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * eps
            recon = self.decoder(z, training=True)

            recon_flat = tf.reshape(recon, [tf.shape(recon)[0], -1])
            data_flat = tf.reshape(data, [tf.shape(data)[0], -1])
            
            # Reconstruction loss: sum over pixels, mean over batch
            recon_loss_per_sample = tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data_flat, recon_flat),
                axis=-1
            )
            recon_loss = tf.reduce_mean(recon_loss_per_sample)
            
            # KL loss
            kl_loss_per_sample = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 
                axis=-1
            )
            kl_loss = tf.reduce_mean(kl_loss_per_sample)
            
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon": self.recon_loss_tracker.result(),
            "kl": self.kl_loss_tracker.result()
        }
    
    def test_step(self, data):
        """Evaluation step for VAE"""
        # Get batch size
        batch = tf.shape(data)[0]
        
        # Encode
        z_mean, z_log_var = self.encoder(data, training=False)
        
        # Sample
        eps = tf.random.normal(shape=(batch, self.encoder.output[0].shape[-1]))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        
        # Decode
        recon = self.decoder(z, training=False)
        
        # Compute losses
        recon_flat = tf.reshape(recon, [batch, 784])
        data_flat = tf.reshape(data, [batch, 784])
        recon_loss = tf.keras.losses.binary_crossentropy(data_flat, recon_flat)
        recon_loss = tf.reduce_mean(tf.reduce_sum(recon_loss, axis=-1))
        
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        
        total_loss = recon_loss + kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'recon': self.recon_loss_tracker.result(),
            'kl': self.kl_loss_tracker.result()
        }


    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return self.decoder(z)