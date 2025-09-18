import os
import glob
import numpy as np
import tensorflow as tf

def read_from_csv(path):
    """
    Placeholder for your CSV reader.
    Must return np.array of shape (T, N, 2) or (T, features).
    """
    return np.loadtxt(path, delimiter=",")

def loader(noise_dir, pulse_dir, mixed_dir, split_r=0.8, batch_size=32, shuffle=True):
    """
    Load matched CSV triplets (noise, pulse, mixed) into tf.data.Dataset
    
    Args:
        noise_dir, pulse_dir, mixed_dir: str, directories containing *_noise.csv, *_pulse.csv, *_mixed.csv
        split_r: float, train split ratio
        batch_size: int
        shuffle: bool
    
    Returns:
        train_ds, test_ds: tf.data.Dataset
    """
    # Find all *_noise.csv
    noise_files = sorted(glob.glob(os.path.join(noise_dir, "*_noise.csv")))
    
    triplets = []
    for nf in noise_files:
        prefix = os.path.basename(nf).replace("_noise.csv", "")
        pf = os.path.join(pulse_dir, prefix + "_pulse.csv")
        mf = os.path.join(mixed_dir, prefix + "_mixed.csv")
        
        if os.path.exists(pf) and os.path.exists(mf):
            noise = read_from_csv(nf)
            pulse = read_from_csv(pf)
            mixed = read_from_csv(mf)
            triplets.append((noise, pulse, mixed))

    if not triplets:
        raise ValueError("No matched triplets found.")

    # Convert to numpy arrays
    noise_arr = np.array([t[0] for t in triplets])
    pulse_arr = np.array([t[1] for t in triplets])
    mixed_arr = np.array([t[2] for t in triplets])

    # Train/test split
    n_total = len(triplets)
    n_train = int(split_r * n_total)

    train_noise, test_noise = noise_arr[:n_train], noise_arr[n_train:]
    train_pulse, test_pulse = pulse_arr[:n_train], pulse_arr[n_train:]
    train_mixed, test_mixed = mixed_arr[:n_train], mixed_arr[n_train:]

    # Build tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_mixed, (train_pulse, train_noise)))
    test_ds = tf.data.Dataset.from_tensor_slices((test_mixed, (test_pulse, test_noise)))

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=n_train)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"[Loaded] {n_train} train, {n_total - n_train} test samples")
    return train_ds, test_ds
