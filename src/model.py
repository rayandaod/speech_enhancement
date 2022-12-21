import tensorflow as tf

from params import *


class MockEncoder(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MockEncoder, self).__init__(*args, **kwargs)

    def call(self, x):
        return x

    def get_config(self):
        return {}


class MockDecoder(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MockDecoder, self).__init__(*args, **kwargs)

    def call(self, x):
        return x

    def get_config(self):
        return {}


class Encoder(tf.keras.layers.Layer):
    def __init__(self, frame_len, hop_len, fft_size, pad_end=True,
                 *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_size = fft_size
        self.pad_end = pad_end

    def call(self, x):
        return tf.signal.stft(x,
                              frame_length=self.frame_len,
                              frame_step=self.hop_len,
                              fft_length=self.fft_size,
                              window_fn=tf.signal.hann_window,
                              pad_end=self.pad_end,
                              name='stft')

    def get_config(self):
        return {"frame_len": self.frame_len,
                "hop_len": self.hop_len,
                "fft_size": self.fft_size,
                "pad_end": self.pad_end}


class Decoder(tf.keras.layers.Layer):
    def __init__(self, frame_len, hop_len, fft_size, pad_end=True,
                 n_samples=-1, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_size = fft_size
        self.pad_end = pad_end
        self.n_samples = n_samples

    def call(self, x):
        win_fn = tf.signal.inverse_stft_window_fn(frame_step=self.hop_len,
                                                  forward_window_fn=tf.signal.hann_window)
        return tf.signal.inverse_stft(x,
                                    frame_length=self.frame_len,
                                    frame_step=self.hop_len,
                                    fft_length=self.fft_size,
                                    window_fn=win_fn)[:, :self.n_samples]

    def get_config(self):
        return {"frame_len": self.frame_len,
                "hop_len": self.hop_len,
                "fft_size": self.fft_size,
                "pad_end": self.pad_end,
                "n_samples": self.n_samples}


def mock_encoder_decoder():
    inputs = tf.keras.Input(shape=(N_SAMPLES,)) # (None, 64000, 1)
    mock_encoder = MockEncoder(name='MockEncoder')
    mock_decoder = MockDecoder(name='MockDecoder')

    x = mock_encoder(inputs)
    x = mock_decoder(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def stft_istft_model():
    inputs = tf.keras.Input(shape=(N_SAMPLES,)) # (None, 64000)
    encoder = Encoder(frame_len=FRAME_LEN, hop_len=HOP_LEN, fft_size=FFT_SIZE,
                      pad_end=PAD_END, name='Encoder')
    decoder = Decoder(frame_len=FRAME_LEN, hop_len=HOP_LEN, fft_size=FFT_SIZE,
                      pad_end=PAD_END, n_samples=N_SAMPLES, name='Decoder')

    x = encoder(inputs)
    x = decoder(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    mock_encoder_decoder().summary()
    stft_istft_model().summary()