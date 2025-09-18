# Load datasets
train_ds, test_ds = loader(
    noise_dir="data/noise",
    pulse_dir="data/pulse",
    mixed_dir="data/mixed",
    split_r=0.8,
    batch_size=16
)

# Build base model
base_model = build_pulse_motion_unet_with_latent(T=128)

# Wrap in trainer (with orthogonality regularizer)
trainer = PulseMotionTrainer(base_model, lambda_ortho=0.1)
trainer.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss_fn=lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + 1e-8)
)

# Train using datasets, not numpy arrays
history = trainer.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50
)

# ---------------------------
# Visualization (sanity check)
# ---------------------------
# Take one example from test dataset
for noisy_batch, (pulse_batch, motion_batch) in test_ds.take(1):
    noisy_example = noisy_batch[0].numpy()     # (T, N, 2)
    pulse_true = pulse_batch[0].numpy()
    motion_true = motion_batch[0].numpy()
    break

# Predict
pulse_pred, motion_pred, _, _ = base_model.predict(noisy_example[None, ...])
pulse_pred = pulse_pred[0]   # (T, N, 2)
motion_pred = motion_pred[0]

# Save comparison videos
os.makedirs("results", exist_ok=True)
save_comparison_video(pulse_true, pulse_pred, "results/pulse_compare.avi", title="Pulse")
save_comparison_video(motion_true, motion_pred, "results/motion_compare.avi", title="Motion")
