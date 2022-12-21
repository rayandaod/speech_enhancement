import tensorflow as tf

from params import *
from data import *

@tf.function
def main():
    # # Load a mixture
    # x = tf.constant('/Users/rayandaod/Documents/Docs/Learning/bose_training/logs/mixture_0/mixture.wav')
    # x = tf.io.read_file(x)
    # x, sr = tf.audio.decode_wav(x, desired_channels=N_CHANNELS)

    # Create a batch of sample tensors
    x = tf.random.uniform(shape=[BATCH_SIZE, N_SAMPLES], minval=-1, maxval=1, dtype=tf.float32)

    #print(x.shape)
    #x = tf.squeeze(x, axis=-1)
    print(x.shape)
    # Apply the stft function
    x_stft = tf.signal.stft(x,
                          frame_length=FRAME_LEN,
                          frame_step=HOP_LEN,
                          fft_length=FFT_SIZE,
                          window_fn=tf.signal.hann_window,
                          pad_end=True,
                          name='stft')
    # Print the shape of the stft tensor
    print(x_stft.shape)


if __name__ == '__main__':
    main()