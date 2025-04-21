import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Lambda
from cluster_center_generation import (
    compute_sigma2_per_class,
    compute_class_means,
    compute_class_distance_matrix,
    find_initial_radius,
    random_points_on_sphere,
    update_positions,
    optimize_radius,
    assign_labels_to_clusters
)
import numpy as np

class LCVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, n_classes, lambda_kl=0.05, k=40, batch_size=128):
        super(LCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.lambda_kl = lambda_kl
        self.k = k
        self.batch_size = batch_size

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

        # Placeholders for cluster centers and variances
        self.cluster_centers = None
        self.sigma2_per_class = None

    def build_clustering(self, x, y):
        n_clusters = self.n_classes
        sigma2_dict = compute_sigma2_per_class(x, y)
        sigma2 = max(sigma2_dict.values())
        sigma = np.sqrt(sigma2)
        alpha = self.k * sigma

        # Class-based variances
        initial_variance_vector = np.array([sigma2_dict[c] for c in range(n_clusters)], dtype=np.float32)
        initial_variance_matrix = np.tile(initial_variance_vector[:, np.newaxis], (1, self.latent_dim))
        self.sigma2_per_class = self.add_weight(
            name="sigma2_per_class",
            shape=initial_variance_matrix.shape,
            initializer=tf.constant_initializer(initial_variance_matrix),
            trainable=True,
            dtype=tf.float32
        )

        # Affichage des sigma initiaux par classe
        initial_sigma_values = np.sqrt(initial_variance_vector)
        print("\n🔹 Sigma initial par classe :")
        for i, sigma in enumerate(initial_sigma_values):
            print(f"    Classe {i} — σ init = {sigma:.4f}")
        print(f"    ➤ σ moyen global : {initial_sigma_values.mean():.4f}\n")

        # Cluster center initialization
        class_means = compute_class_means(x, y)
        class_dist_matrix = compute_class_distance_matrix(class_means)
        R_init = find_initial_radius(n_clusters, self.latent_dim, alpha)
        initial_points = random_points_on_sphere(n_clusters, self.latent_dim, 1.0)
        optimized_points = update_positions(initial_points, 1.0)
        optimized_points *= R_init
        final_R, optimized_points = optimize_radius(optimized_points, R_init, alpha)
        cluster_mapping = assign_labels_to_clusters(class_dist_matrix, optimized_points)
        ordered_centers = np.array([cluster_mapping[i] for i in sorted(cluster_mapping.keys())])
        self.cluster_centers = tf.constant(ordered_centers.astype(np.float32))

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
        return mu, logvar

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def compute_loss(self, x, y):
        x_hat, mu, logvar, _ = self.call(x)
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
        recon_loss = tf.reduce_sum(bce_loss_fn(x, x_hat), axis=-1)

        # KL divergence
        sigma2_q = tf.exp(logvar)
        # y: [batch_size], entiers dans [0, n_classes)
        one_hot_y = tf.one_hot(y, depth=self.n_classes)  # [batch_size, n_classes]

        # Produits matriciels différentiables
        c_y = tf.matmul(one_hot_y, self.cluster_centers)         # [batch_size, latent_dim]
        sigma2_p = tf.matmul(one_hot_y, self.sigma2_per_class)   # [batch_size, latent_dim]


        kl = 0.5 * tf.reduce_sum(
            tf.math.log(sigma2_p + 1e-8) - tf.math.log(sigma2_q + 1e-8)
            + sigma2_q / (sigma2_p + 1e-8)
            + tf.square(mu - c_y) / (sigma2_p + 1e-8)
            - 1.0,
            axis=1
        )

        total_loss = tf.reduce_mean(recon_loss + self.lambda_kl * kl)
        return total_loss, tf.reduce_mean(recon_loss), tf.reduce_mean(kl)

    @tf.function
    def train_step(self, x_batch, y_batch, optimizer):
        with tf.GradientTape() as tape:
            loss, recon, kl = self.compute_loss(x_batch, y_batch)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, recon, kl

    def train(self, train_dataset, optimizer, epochs=20):
        from tqdm import tqdm
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            print(f"\nEpoch {epoch+1}/{epochs}")
            bar = tqdm(train_dataset, desc="Batch", leave=False)

            for x_batch, y_batch in bar:
                # Ignore the batch if it's smaller than the expected batch size
                if tf.shape(x_batch)[0] < self.batch_size:
                    continue

                loss, recon, kl = self.train_step(x_batch, y_batch, optimizer)
                total_loss += loss.numpy()
                total_recon += recon.numpy()
                total_kl += kl.numpy()

                bar.set_postfix({
                    "Loss": f"{loss.numpy():.2f}",
                    "Recon": f"{recon.numpy():.2f}",
                    "KL": f"{kl.numpy():.2f}"
                })

            print(f"→ Epoch {epoch+1:02d} | Total Loss: {total_loss:.1f} | Recon: {total_recon:.1f} | KL: {total_kl:.1f}")
            print(f"→ Epoch {epoch+1:02d} | Total Loss: {total_loss:.1f} | Recon: {total_recon:.1f} | KL: {total_kl:.1f}")

            # Moyenne des sigma par classe
            sigma2_values = self.sigma2_per_class.numpy()
            sigma_values = np.sqrt(sigma2_values)
            mean_sigma_per_class = sigma_values.mean(axis=1)
            global_mean_sigma = mean_sigma_per_class.mean()

            for i, sigma_c in enumerate(mean_sigma_per_class):
                print(f"    Classe {i} — σ moyen = {sigma_c:.4f}")
            print(f"    ➤ σ moyen global : {global_mean_sigma:.4f}")

