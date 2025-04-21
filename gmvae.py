import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class GMVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, n_classes, lambda_kl=0.05):
        super(GMVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.lambda_kl = lambda_kl

        # Encoder
        self.enc_dense1 = Dense(1024, activation="relu")
        self.bn1 = BatchNormalization()
        self.do1 = Dropout(0.3)

        self.enc_dense2 = Dense(512, activation="relu")
        self.bn2 = BatchNormalization()
        self.do2 = Dropout(0.3)

        self.enc_dense3 = Dense(256, activation="relu")
        self.bn3 = BatchNormalization()
        self.do3 = Dropout(0.3)

        self.enc_dense4 = Dense(128, activation="relu")
        self.bn4 = BatchNormalization()

        self.fc_mu = Dense(latent_dim)
        self.fc_logvar = Dense(latent_dim)
        self.fc_class = Dense(n_classes)  # logits for q(y|x)

        # Decoder
        self.dec_dense1 = Dense(128, activation="relu")
        self.dec_bn1 = BatchNormalization()
        self.dec_dense2 = Dense(256, activation="relu")
        self.dec_do2 = Dropout(0.1)
        self.dec_dense3 = Dense(512, activation="relu")
        self.dec_bn3 = BatchNormalization()
        self.dec_dense4 = Dense(1024, activation="relu")
        self.dec_bn4 = BatchNormalization()
        self.dec_out = Dense(input_dim, activation="sigmoid")

        # Class priors
        self.prior_mu = tf.Variable(tf.random.normal([n_classes, latent_dim]), trainable=True)
        self.prior_logvar = tf.Variable(tf.zeros([n_classes, latent_dim]), trainable=True)

    def encode(self, x):
        x = self.enc_dense1(x)
        x = self.bn1(x)
        x = self.do1(x)

        x = self.enc_dense2(x)
        x = self.bn2(x)
        x = self.do2(x)

        x = self.enc_dense3(x)
        x = self.bn3(x)
        x = self.do3(x)

        x = self.enc_dense4(x)
        x = self.bn4(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logits = self.fc_class(x)

        return mu, logvar, logits

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * logvar)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_dense1(z)
        x = self.dec_bn1(x)
        x = self.dec_dense2(x)
        x = self.dec_do2(x)
        x = self.dec_dense3(x)
        x = self.dec_bn3(x)
        x = self.dec_dense4(x)
        x = self.dec_bn4(x)
        return self.dec_out(x)

    def call(self, x):
        mu, logvar, logits = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, logits, z

    def compute_loss(self, x, y_true):
        x_hat, mu, logvar, logits, _ = self.call(x)

        recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_hat))
        class_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

        mu_y = tf.gather(self.prior_mu, y_true)
        logvar_y = tf.gather(self.prior_logvar, y_true)

        kl = 0.5 * tf.reduce_sum(
            logvar_y - logvar +
            (tf.exp(logvar) + tf.square(mu - mu_y)) / tf.exp(logvar_y) - 1
        )

        total = recon_loss + class_loss + self.lambda_kl * kl
        return total, recon_loss, class_loss, kl

    @tf.function
    def train_step(self, x_batch, y_batch, optimizer):
        with tf.GradientTape() as tape:
            loss, recon, classif, kl = self.compute_loss(x_batch, y_batch)
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, recon, classif, kl

    def train(self, train_dataset, optimizer, epochs=20):
        """
        Train the GMVAE model for a given number of epochs.

        Args:
            train_dataset (tf.data.Dataset): Dataset containing (x, y) pairs.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to use.
            epochs (int): Number of training epochs.
        """
        from tqdm import tqdm
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_class = 0
            total_kl = 0

            print(f"\nEpoch {epoch+1}/{epochs}")
            batch_bar = tqdm(train_dataset, desc="Batch", leave=False)

            for x_batch, y_batch in batch_bar:
                loss, recon, classif, kl = self.train_step(x_batch, y_batch, optimizer)

                total_loss += loss.numpy()
                total_recon += recon.numpy()
                total_class += classif.numpy()
                total_kl += kl.numpy()

                batch_bar.set_postfix({
                    "Loss": f"{loss.numpy():.2f}",
                    "Recon": f"{recon.numpy():.2f}",
                    "Class": f"{classif.numpy():.2f}",
                    "KL": f"{kl.numpy():.2f}"
                })

            print(f"→ Epoch {epoch+1:02d} | Total Loss: {total_loss:.1f} | Recon: {total_recon:.1f} | Class: {total_class:.1f} | KL: {total_kl:.1f}")
