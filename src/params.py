LIBRISPEECH_DEV_PATH = '/Users/rayandaod/Documents/Docs/Learning/bose_training/data/librispeech/dev/dev-clean/**/**/*.wav'
LIBRISPEECH_TRAIN_PATH = '/Users/rayandaod/Documents/Docs/Learning/bose_training/data/librispeech/train/train-clean-100/**/**/*.wav'
LIBRISPEECH_TEST_PATH = '/Users/rayandaod/Documents/Docs/Learning/bose_training/data/librispeech/test/test-clean/**/**/*.wav'

NOISE_PATH = '/Users/rayandaod/Documents/Docs/Learning/bose_training/data/REVERB_chall_noises/**/*.wav'

LOGS_PATH = '/Users/rayandaod/Documents/Docs/Learning/bose_training/logs'

SEED = 42
SAMPLE_RATE = 16000
DURATION = 4
N_SAMPLES = SAMPLE_RATE * DURATION
N_CHANNELS = 1  # Mono (1). To change to stereo (2), change the stft function and input layer in model.py

SNR_MIN = -10
SNR_MAX = 0

BATCH_SIZE = 8
N_BATCHES_PER_EPOCH = 100
MAX_EPOCHS = 10

FRAME_LEN = 512
HOP_LEN = 256
FFT_SIZE = 512
PAD_END = True