import tensorflow as tf

from params import *
from model import stft_istft_model
from data import prepare_ds


@tf.function
def _train_step(mixture, clean_speech, model, optimizer, loss):
        with tf.GradientTape() as tape:
            prediction = model(mixture)
            loss = loss(clean_speech, prediction)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss


def train_loop(ds, model, optimizer, loss):
    epoch = 1
    loss_value = 0
    for t_step, (mixture, clean_speech, _, _) in enumerate(ds, start=1):
        loss_value += _train_step(mixture, clean_speech, model, optimizer, loss)
        
        # Print training metrics every 10 batches
        if t_step % 10 == 0:
            print(f"Epoch {epoch} (Step {t_step})  |  loss {loss_value/t_step % N_BATCHES_PER_EPOCH:.4f}", end='\r')
            
        # End of an "epoch" as defined by N_BATCHES_PER_EPOCH
        if t_step % N_BATCHES_PER_EPOCH == 0:
            # Print training metrics without \r to add \n at the end
            print(f"Epoch {epoch} (Step {t_step})  |  loss {loss_value/t_step%N_BATCHES_PER_EPOCH:.4f}")

            # Validation step
            pass
            
            # Maximum number of epochs?
            if epoch == MAX_EPOCHS:
                print('Maximum number of epochs reached.')
                break

            epoch += 1
            loss_value = 0


if __name__ == '__main__':
    ds = prepare_ds(stage='train')
    model = stft_istft_model()
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.MeanAbsoluteError(name='mae')
    train_loop(ds, model, optimizer, loss)

    