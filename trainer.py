class PulseMotionTrainer(tf.keras.Model):
    def __init__(self, base_model, lambda_ortho=0.1):
        super().__init__()
        self.base_model = base_model
        self.lambda_ortho = lambda_ortho

        # Trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pulse_loss_tracker = tf.keras.metrics.Mean(name="pulse_loss")
        self.motion_loss_tracker = tf.keras.metrics.Mean(name="motion_loss")
        self.ortho_loss_tracker = tf.keras.metrics.Mean(name="ortho_loss")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, (y_pulse, y_motion) = data

        with tf.GradientTape() as tape:
            pulse_pred, motion_pred, pulse_latent, motion_latent = self.base_model(x, training=True)

            # Reconstruction losses
            loss_pulse = self.loss_fn(y_pulse, pulse_pred)
            loss_motion = self.loss_fn(y_motion, motion_pred)

            # Orthogonality regularizer
            loss_ortho = orthogonality_loss(pulse_latent, motion_latent)

            # Total
            loss = loss_pulse + loss_motion + self.lambda_ortho * loss_ortho

        # Backprop
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        # Update trackers
        self.loss_tracker.update_state(loss)
        self.pulse_loss_tracker.update_state(loss_pulse)
        self.motion_loss_tracker.update_state(loss_motion)
        self.ortho_loss_tracker.update_state(loss_ortho)

        return {
            "loss": self.loss_tracker.result(),
            "pulse_loss": self.pulse_loss_tracker.result(),
            "motion_loss": self.motion_loss_tracker.result(),
            "ortho_loss": self.ortho_loss_tracker.result(),
        }
