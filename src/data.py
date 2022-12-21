import tensorflow as tf
import pprint
import os

import helper

from params import *

pp = pprint.PrettyPrinter(width=41, compact=True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_and_decode(filename):
    audio_file = tf.io.read_file(filename, name='read_file')
    audio_data, sr = tf.audio.decode_wav(audio_file, desired_channels=N_CHANNELS, name='decode_wav')
    audio_data = tf.squeeze(audio_data, axis=-1, name='squeeze_audio')
    metadata = {
        'file_path': filename,
        'sample_rate': sr,
        'shape': tf.shape(audio_data)
    }
    return audio_data, metadata

def pad_or_truncate(audio_decoded, metadata):
    start = tf.constant(0, dtype=tf.int32)
    n_samples = tf.shape(audio_decoded)[0]
    n_samples_objective = SAMPLE_RATE * DURATION
    if n_samples <= n_samples_objective:
        audio_decoded = tf.pad(audio_decoded, [[0, n_samples_objective - n_samples]], name='pad_audio')
    else:
        # Randomly select a segment of the audio
        start = tf.random.uniform(shape=[], minval=0, maxval=n_samples - n_samples_objective, dtype=tf.int32, name='random_start')
        audio_decoded = audio_decoded[start:start + n_samples_objective]
    
    metadata['duration'] = DURATION
    metadata['n_samples'] = n_samples_objective
    metadata['start'] = start
    
    return audio_decoded, metadata

def normalise(audio_decoded, metadata):
    max_value = tf.reduce_max(tf.abs(audio_decoded))
    normalised_audio = tf.divide(audio_decoded, max_value)

    metadata['max_value'] = max_value
    
    return normalised_audio, metadata

def create_mixture(clean_speech, noise):
    clean_speech, clean_speech_metadata = clean_speech
    noise, noise_metadata = noise

    # Randomly choose a SNR level
    snr = tf.random.uniform(shape=[], minval=SNR_MIN, maxval=SNR_MAX, dtype=tf.float32)

    # Add the noise to the speech with the specified SNR
    clean_speech_loudness = tf.reduce_mean(tf.square(clean_speech))
    desired_noise_loudness = tf.sqrt(clean_speech_loudness / (10 ** (snr / 10)))
    new_noise = tf.multiply(noise, desired_noise_loudness)
    mixture = tf.add(clean_speech, new_noise)

    metadata = {
        'clean_speech_metadata': clean_speech_metadata,
        'noise_metadata': noise_metadata,
        'SNR': snr
    }

    return mixture, clean_speech, new_noise, metadata


def prepare_ds(stage='train'):
    # Set the data path
    if stage == 'train':
        DATA_PATH = LIBRISPEECH_TRAIN_PATH
    elif stage == 'dev':
        DATA_PATH = LIBRISPEECH_DEV_PATH
    elif stage == 'test':
        DATA_PATH = LIBRISPEECH_TEST_PATH
    else:
        raise ValueError('Invalid stage name')

    # Create the datasets
    clean_speech_ds = tf.data.Dataset.list_files(DATA_PATH, seed=SEED)
    noise_ds = tf.data.Dataset.list_files(NOISE_PATH, seed=SEED)

    # Shuffle the datasets
    clean_speech_ds = clean_speech_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True, seed=SEED)
    noise_ds = noise_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True, seed=SEED)

    # Read, decode, and pad/truncate the audio files
    clean_speech_ds = clean_speech_ds.map(read_and_decode, num_parallel_calls=AUTOTUNE)
    noise_ds = noise_ds.map(read_and_decode, num_parallel_calls=AUTOTUNE)

    # Pad or truncate the audio files
    clean_speech_ds = clean_speech_ds.map(pad_or_truncate, num_parallel_calls=AUTOTUNE)
    noise_ds = noise_ds.map(pad_or_truncate, num_parallel_calls=AUTOTUNE)

    # Normalise the audio files
    clean_speech_ds = clean_speech_ds.map(normalise, num_parallel_calls=AUTOTUNE)
    noise_ds = noise_ds.map(normalise, num_parallel_calls=AUTOTUNE)

    # Repeat both datasets
    clean_speech_ds = clean_speech_ds.repeat()
    noise_ds = noise_ds.repeat()

    # Zip the datasets together
    ds = tf.data.Dataset.zip((clean_speech_ds, noise_ds))

    # Create mixture
    ds = ds.map(create_mixture, num_parallel_calls=AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(BATCH_SIZE)

    # Prefetch the dataset
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if __name__ == '__main__':
    ds = prepare_ds()
    ds = ds.unbatch()
    ds_iterator = tf.compat.v1.data.make_one_shot_iterator(ds)

    mixture_name = 'mixture'
    clean_speech_name = 'clean_speech'
    noise_name = 'noise'
    metadata_name = 'metadata'

    for i in range(5):
        mixture_b, clean_speech_b, noise_b, metadata_b = ds_iterator.get_next()
        
        # create a folder to store the audio files
        folder_name = os.path.join(LOGS_PATH, f'{mixture_name}_{i}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # save the mixture
        mixture_b = tf.expand_dims(mixture_b, axis=-1, name='expand_dims')
        encoded_audio_data = tf.audio.encode_wav(mixture_b, SAMPLE_RATE, name='encode_wav')
        write_file_op = tf.io.write_file(os.path.join(folder_name, f'{mixture_name}.wav'), encoded_audio_data)

        # save the clean speech
        clean_speech_b = tf.expand_dims(clean_speech_b, axis=-1, name='expand_dims')
        encoded_audio_data = tf.audio.encode_wav(clean_speech_b, SAMPLE_RATE, name='encode_wav')
        write_file_op = tf.io.write_file(os.path.join(folder_name, f'{clean_speech_name}.wav'), encoded_audio_data)

        # save the noise
        noise_b = tf.expand_dims(noise_b, axis=-1, name='expand_dims')
        encoded_audio_data = tf.audio.encode_wav(noise_b, SAMPLE_RATE, name='encode_wav')
        write_file_op = tf.io.write_file(os.path.join(folder_name, f'{noise_name}.wav'), encoded_audio_data)

        # convert the metadata to a dictionary of strings
        metadata_b = helper.dict_of_tensors_to_dict_of_strings(metadata_b)
        
        # save the metadata
        with open(os.path.join(folder_name, f"{metadata_name}.txt"), "w") as f:
            f.write(pprint.pformat(metadata_b, indent=4, width=250))