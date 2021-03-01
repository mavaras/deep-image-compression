# Standard libraries
from argparse import ArgumentParser
from typing import Optional

# Third party libraries
import tensorflow as tf
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import Mean

# Local libraries
from model.cae import CAE


def train(
    model: CAE,
    output_path: str,
    epochs: Optional[int],
    image_width: int,
    image_height: int,
    log_freq: int,
    save_freq: int
) -> None:
    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            pred_image = model(image)
            model_trainable_variables = model.trainable_variables
            loss = MSE(image, pred_image)
            gradients = tape.gradient(loss, model_trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_trainable_variables))
            train_loss(loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, transformer=model)
    manager = tf.train.CheckpointManager(ckpt, output_path, max_to_keep=1)
    train_loss = Mean(name='train_loss')

    epochs = epochs or len(dataset)
    section_size = 128
    for step, train_image in enumerate(dataset):
        train_image = train_image.numpy()
        for c in range(image_height / section_size):
            for j in range(image_width / section_size):
                cc = section_size * c
                jj = section_size * j
                train_image_batch = train_image[:, cc : cc + section_size, jj : jj + section_size, :]
                train_image_tensor = tf.convert_to_tensor(train_image_batch)
                train_step(train_image_tensor)

        if step % log_freq == 0:
            print(
                f'Step {step}/{epochs}, '
                f'Loss: {train_loss.result()}, '
            )
        if step % save_freq == 0 or step == epochs - 1:
            print(f'Saved checkpoint: {manager.save()}')
            train_loss.reset_states()

            if epochs and step == epochs:
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output-path',
        required=False,
        default='trained_models/',
        type=str
    )
    parser.add_argument('--epochs', required=False, default=None, type=int)
    parser.add_argument(
        '--image-width',
        required=False,
        default=1280,
        type=int
    )
    parser.add_argument(
        '--image-height',
        required=False,
        default=720,
        type=int
    )
    parser.add_argument('--log-freq', required=False, default=5, type=int)
    parser.add_argument('--save-freq', required=False, default=15, type=int)
    args = parser.parse_args()

    model = CAE()
    train(
        model,
        output_path=args.output_path,
        image_height=args.image_height,
        image_width=args.image_width,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        epochs=epochs
    )
    