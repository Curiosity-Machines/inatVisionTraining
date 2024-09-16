import os
import time
import pandas as pd
import numpy as np
import argparse
import yaml

import json

import tensorflow as tf
from tensorflow import keras
# import wandb
# from wandb.integration.keras import WandbMetricsLogger

AUTOTUNE = tf.data.AUTOTUNE
tf.config.optimizer.set_jit(True) 

from datasets import inat_dataset
from nets import nets
from dynamic_dropout import DynamicDropout

class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        lr = self.model.optimizer.learning_rate
        print("learning rate is {}".format(lr))
        # wandb.log({"lr": "{}".format(lr)}, commit=False)


def make_training_callbacks(config, iteration, model):
    checkpoint_file_name = f"checkpoint-{iteration}-{{epoch:02d}}.weights.h5"

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=config["TENSORBOARD_LOG_DIR"],
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq=20,
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata={},
            write_steps_per_second=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["CHECKPOINT_DIR"], checkpoint_file_name),
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=config["BACKUP_DIR"],
        ),
        # WandbMetricsLogger(log_freq=config["WANDB_LOG_FREQ"]),
        LRLogger(),
    ]

    return callbacks


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Train an iNat model.")
    parser.add_argument(
        "--config_file", required=True, help="YAML config file for training."
    )
    args = parser.parse_args()

    # read in config file
    if not os.path.exists(args.config_file):
        print("No config file.")
        return
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # wandb.init(
    #     project=config["WANDB_PROJECT"],
    #     config=config
    # )

    if config["TRAIN_MIXED_PRECISION"]:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    if config["MULTIGPU"]:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # Load label_to_index if exists
    label_to_index = None

    # Phase 1: Progressive Training (Small Images, Minimal Dropout, No Augmentation)
    with strategy.scope():
        sizes = config.get("SIZES") 
        magnitudes = config.get("AUGMENT_MAGNITUDES")
        dropouts = config.get("DROPOUTS")
        last_size = sizes[len(sizes) - 1]
        last_checkpoint = None

        remaining_epochs = config["NUM_EPOCHS"]
        epochs_per_iteration = int(remaining_epochs / len(sizes))
        iteration = 0

        val_ds = None
        num_val_examples = 0

        while remaining_epochs > 0:
            size = sizes[iteration]
            magnitude = magnitudes[iteration]
            dropout = dropouts[iteration]
            epoch = epochs_per_iteration * iteration

            print(f"Training iteration {iteration} with {size}x{size}, augment: {magnitude}, dropout: {dropout} for {epochs_per_iteration} epochs, starting from {last_checkpoint}")

            # Create optimizer
            optimizer = keras.optimizers.RMSprop(
                learning_rate=config["INITIAL_LEARNING_RATE"],
                rho=config["RMSPROP_RHO"],
                momentum=config["RMSPROP_MOMENTUM"],
                epsilon=config["RMSPROP_EPSILON"],
                weight_decay=1e-5,
            )

            # Create neural network with minimal dropout
            model = nets.make_neural_network(
                base_arch_name=config["MODEL_NAME"],
                weights=config["PRETRAINED_MODEL"],
                image_size=[size, size],
                n_classes=config["NUM_CLASSES"],
                input_dtype=tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
                ckpt=None,
                train_full_network=config["TRAIN_FULL_MODEL"],
                dropout=dropout,
                factorize=config["FACTORIZE_FINAL_LAYER"] if "FACTORIZE_FINAL_LAYER" in config else False,
                fact_rank=config["FACT_RANK"] if "FACT_RANK" in config else None
            )

            if last_checkpoint != None:
                model.load_weights(last_checkpoint)

            # Define loss
            if config["DO_LABEL_SMOOTH"]:
                if config["LABEL_SMOOTH_MODE"] == "flat":
                    loss = tf.keras.losses.CategoricalCrossentropy(
                        label_smoothing=config["LABEL_SMOOTH_PCT"]
                    )
                else:
                    assert False, "Unsupported label smoothing mode."
            else:
                loss = tf.keras.losses.CategoricalCrossentropy()

            # Compile the network for training
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=[
                    "accuracy",
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="top10_accuracy"),
                ],
            )

            # Setup callbacks
            training_callbacks = make_training_callbacks(config, iteration, model)

            (train_ds, num_train_examples, label_to_index) = inat_dataset.make_dataset(
                config["TRAINING_DATA"],
                label_column_name=config["LABEL_COLUMN_NAME"],
                image_size=(size,size),
                batch_size=config["BATCH_SIZE"],
                shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
                repeat_forever=True,
                augment_magnitude=magnitude,
                label_to_index=label_to_index
            )

            STEPS_PER_EPOCH = int(np.ceil(num_train_examples / config["BATCH_SIZE"]))

            (val_ds, num_val_examples, label_to_index) = inat_dataset.make_dataset(
                config["VAL_DATA"],
                label_column_name=config["LABEL_COLUMN_NAME"],
                image_size=(size, size),
                batch_size=config["BATCH_SIZE"],
                shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
                repeat_forever=True,
                augment_magnitude=0.0,
                label_to_index=label_to_index
            )

            # Training & val step counts
            VAL_IMAGE_COUNT = (
                config["VALIDATION_PASS_SIZE"]
                if config["VALIDATION_PASS_SIZE"] is not None
                else num_val_examples
            )

            VAL_STEPS = int(np.ceil(VAL_IMAGE_COUNT / config["BATCH_SIZE"]))

            model.fit(
                train_ds,
                validation_data=val_ds,
                validation_steps=VAL_STEPS,
                epochs=epochs_per_iteration,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=training_callbacks,
            )

            remaining_epochs -= epochs_per_iteration

            iteration += 1

            # Checkpoint dir gets wiped with each iteration
            last_checkpoint = os.path.join(config["CHECKPOINT_DIR"], f"checkpoint-{iteration - 1}-{epochs_per_iteration:02d}.weights.h5")

        # Save the final model
        save_dir = CONFIG["FINAL_SAVE_DIR"]
        model.save(f"{save_dir}/final.h5")

    # wandb.finish()


if __name__ == "__main__":
    main()
