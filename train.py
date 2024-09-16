import os
import time
import pandas as pd
import numpy as np
import argparse
import yaml

import json

import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger

AUTOTUNE = tf.data.AUTOTUNE
tf.config.optimizer.set_jit(True) 

from datasets import inat_dataset
from nets import nets
from progressive_training_callbacks import ProgressiveTrainingCallback
from dynamic_dropout import DynamicDropout

class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        lr = self.model.optimizer.learning_rate
        print("learning rate is {}".format(lr))
        wandb.log({"lr": "{}".format(lr)}, commit=False)


def make_training_callbacks(config, model, make_dataset_fn):
    def lr_scheduler_fn(epoch):
        return config["INITIAL_LEARNING_RATE"] * (config["LR_DECAY_FACTOR"] ** (epoch // config["EPOCHS_PER_LR_DECAY"]))

    checkpoint_file_name = "checkpoint-{epoch:02d}-{val_accuracy:.2f}.weights.h5"
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
        WandbMetricsLogger(log_freq=config["WANDB_LOG_FREQ"]),
        LRLogger(),
        ProgressiveTrainingCallback(config, make_dataset_fn),
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

    wandb.init(
        project=config["WANDB_PROJECT"],
        config=config
    )

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
        # Load train & val datasets for Phase 1
        (train_ds_small, num_train_examples, train_labels, label_to_index) = inat_dataset.make_dataset(
            config["TRAINING_DATA"],
            label_column_name=config["LABEL_COLUMN_NAME"],
            image_size=[128, 128],
            batch_size=config["BATCH_SIZE"],
            shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
            repeat_forever=True,
            augment=False,
            label_to_index=label_to_index
        )
        if train_ds_small is None or num_train_examples == 0:
            print("No training dataset for Phase 1.")
            return

        (val_ds, num_val_examples) = inat_dataset.make_dataset(
            config["VAL_DATA"],
            label_column_name=config["LABEL_COLUMN_NAME"],
            image_size=[128, 128],
            batch_size=config["BATCH_SIZE"],
            shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
            repeat_forever=True,
            augment=False,
            label_to_index=label_to_index
        )
        if val_ds is None or num_val_examples == 0:
            print("No validation dataset.")
            return

        # Create optimizer
        STEPS_PER_EPOCH = int(np.ceil(num_train_examples / config["BATCH_SIZE"]))
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["INITIAL_LEARNING_RATE"], amsgrad=True)

        # Create neural network with minimal dropout
        model = nets.make_neural_network(
            base_arch_name=config["MODEL_NAME"],
            weights=config["PRETRAINED_MODEL"],
            image_size=[128, 128],
            n_classes=config["NUM_CLASSES"],
            input_dtype=tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
            train_full_network=config["TRAIN_FULL_MODEL"],
            ckpt=config["CHECKPOINT"] if "CHECKPOINT" in config else None,
            factorize=config["FACTORIZE_FINAL_LAYER"] if "FACTORIZE_FINAL_LAYER" in config else False,
            fact_rank=config["FACT_RANK"] if "FACT_RANK" in config else None,
            dropout_rate=config.get("INITIAL_DROPOUT_RATE", 0.1),
            l2_reg=config["L2_REG"] if "L2_REG" in config else 1e-5,
            augment=False
        )

        if model is None:
            assert False, "No model to train."

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
        training_callbacks = make_training_callbacks(config, model, inat_dataset.make_dataset)

        # Training & val step counts
        VAL_IMAGE_COUNT = (
            config["VALIDATION_PASS_SIZE"]
            if config["VALIDATION_PASS_SIZE"] is not None
            else num_val_examples
        )
        VAL_STEPS = int(np.ceil(VAL_IMAGE_COUNT / config["BATCH_SIZE"]))
        print(
            "{} val steps for {} val pass images of {} total val images.".format(
                VAL_STEPS, VAL_IMAGE_COUNT, num_val_examples
            )
        )

        # Phase 1 Training
        print(f"Starting Phase 1 Training: Epochs 1 to {config.get('PROGRESSIVE_TRAINING_EPOCH', 20)}")
        history_phase1 = model.fit(
            train_ds_small,
            validation_data=val_ds,
            validation_steps=VAL_STEPS,
            epochs=config.get("PROGRESSIVE_TRAINING_EPOCH", 20),
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=training_callbacks,
        )

        # Phase 2: Final Training (Large Images, Augmentation, Standard Dropout)
        # Update Dropout Rate
        for layer in model.layers:
            if isinstance(layer, DynamicDropout):
                print(f"Updating dropout rate from {layer.rate.numpy()} to {config['DROPOUT_PCT']}")
                layer.set_rate(config['DROPOUT_PCT'])

        # Rebuild the training dataset for Phase 2
        print(f"Rebuilding training dataset for Phase 2 with image size {config['IMAGE_SIZE']} and augmentation enabled.")
        (train_ds_large, _, _, _) = inat_dataset.make_dataset(
            config["TRAINING_DATA"],
            label_column_name=config["LABEL_COLUMN_NAME"],
            image_size=config["IMAGE_SIZE"],
            batch_size=config["BATCH_SIZE"],
            shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
            repeat_forever=True,
            augment=True,
            label_to_index=None
        )

        # Calculate remaining epochs
        remaining_epochs = config["NUM_EPOCHS"] - config.get("PROGRESSIVE_TRAINING_EPOCH", 20)
        if remaining_epochs > 0:
            print(f"Starting Phase 2 Training: Epochs {config.get('PROGRESSIVE_TRAINING_EPOCH', 20) + 1} to {config['NUM_EPOCHS']}")
            history_phase2 = model.fit(
                train_ds_large,
                validation_data=val_ds,
                validation_steps=VAL_STEPS,
                epochs=config["NUM_EPOCHS"],
                initial_epoch=config.get("PROGRESSIVE_TRAINING_EPOCH", 20),
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=training_callbacks,
            )
        else:
            print("Total epochs less than or equal to progressive training epoch. Skipping Phase 2.")

        # Save the final model
        model.save(config["FINAL_SAVE_DIR"])

    wandb.finish()


if __name__ == "__main__":
    main()
