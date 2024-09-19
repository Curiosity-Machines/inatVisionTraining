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
from efficientnetv2 import EfficientNetV2_S

class ExponentialMovingAverage(tf.keras.callbacks.Callback):
    def __init__(self, decay):
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)

    def on_train_batch_end(self, batch, logs=None):
        self.ema.apply(self.model.trainable_variables)

    def on_epoch_end(self, epoch, logs=None):
        for var in self.model.trainable_variables:
            var.assign(self.ema.average(var))

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs, total_epochs, decay_rate, decay_steps):
        super(LearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.epoch = 0
        self.lr = initial_lr

    def on_epoch_begin(self, epoch, logs=None):
        epoch = self.epoch # Since this spans multiple models

        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.initial_lr * (self.decay_rate ** ((epoch - self.warmup_epochs) / self.decay_steps))
        print(self.model.optimizer)
        self.model.optimizer.learning_rate.assign(lr)
        print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}.")
        self.epoch += 1
        self.lr = lr

def make_training_callbacks(config, iteration, scheduler):
    checkpoint_file_name = f"checkpoint-{iteration}-{{epoch:02d}}.weights.keras"

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
            save_weights_only=False,
            save_best_only=False,
            monitor="val_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=config["BACKUP_DIR"],
        ),
        # WandbMetricsLogger(log_freq=config["WANDB_LOG_FREQ"]),
        scheduler
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
    model = None

    # Phase 1: Progressive Training (Small Images, Minimal Dropout, No Augmentation)
    with strategy.scope():
        batch_sizes = config.get("BATCH_SIZES") 
        sizes = config.get("SIZES") 
        lrs = config.get("MAX_LRS") 
        magnitudes = config.get("AUGMENT_MAGNITUDES")
        dropouts = config.get("DROPOUTS")
        last_size = sizes[len(sizes) - 1]
        last_checkpoint = None

        remaining_epochs = config["NUM_EPOCHS"]
        epochs_per_iteration = int(remaining_epochs / len(sizes))
        iteration = 0

        val_ds = None
        num_val_examples = 0

        optimizer_weights = None

        while remaining_epochs > 0:
            size = sizes[iteration]
            magnitude = magnitudes[iteration]
            dropout = dropouts[iteration]
            batch_size = batch_sizes[iteration]
            lr = lrs[iteration]
            epoch = epochs_per_iteration * iteration

            print(f"Training iteration {iteration} with {size}x{size}, augment: {magnitude}, lr: {lr}, batch size: {batch_size}, dropout: {dropout} for {epochs_per_iteration} epochs starting from {last_checkpoint}")

            scheduler = LearningRateScheduler(
                initial_lr=lr,
                warmup_epochs=10,  # Adjust based on your warmup duration
                total_epochs=config["NUM_EPOCHS"],
                decay_rate=config["LR_DECAY_FACTOR"],
                decay_steps=config["EPOCHS_PER_LR_DECAY"]
            )

            # Create optimizer
            optimizer = keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=config["RMSPROP_RHO"],
                momentum=config["RMSPROP_MOMENTUM"],
                epsilon=config["RMSPROP_EPSILON"],
                weight_decay=1e-5,
            )

            base_model = EfficientNetV2_S(
                    input_shape=(size, size, 3),
                    pretrained=config.get("PRETRAINED_MODEL", None),
                    classes=config["NUM_CLASSES"],
                    dropout_rate=dropout,
                    include_top=True,
                    final_drop_rate=dropout,
                    weights=config.get("PRETRAINED_MODEL", None),
                    factorize_rank=config["FACT_RANK"] if "FACT_RANK" in config else None
                )

            output = keras.layers.Activation("softmax", dtype="float32", name="predictions")(base_model.output)
            model = keras.Model(inputs=base_model.inputs, outputs=output)
            # model.summary()

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

            # if optimizer_weights != None:
            #     model.optimizer.build(model.trainable_variables)

            #     for var, weight in zip(model.optimizer.variables, optimizer_weights):
            #         var.assign(weight)

            # Setup callbacks
            training_callbacks = make_training_callbacks(config, iteration, scheduler)

            (train_ds, num_train_examples, label_to_index) = inat_dataset.make_dataset(
                config["TRAINING_DATA"],
                label_column_name=config["LABEL_COLUMN_NAME"],
                image_size=(size,size),
                batch_size=batch_size,
                shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
                repeat_forever=True,
                augment_magnitude=magnitude,
                label_to_index=label_to_index
            )

            STEPS_PER_EPOCH = int(np.ceil(num_train_examples / batch_size))

            (val_ds, num_val_examples, label_to_index) = inat_dataset.make_dataset(
                config["VAL_DATA"],
                label_column_name=config["LABEL_COLUMN_NAME"],
                image_size=(size, size),
                batch_size=batch_size,
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

            VAL_STEPS = int(np.ceil(VAL_IMAGE_COUNT / batch_size))

            model.fit(
                train_ds,
                validation_data=val_ds,
                validation_steps=VAL_STEPS,
                epochs=epochs_per_iteration,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=training_callbacks,
            )

            optimizer_weights = [v.numpy() for v in model.optimizer.variables]

            remaining_epochs -= epochs_per_iteration

            iteration += 1

            # Checkpoint dir gets wiped with each iteration
            last_checkpoint = os.path.join(config["CHECKPOINT_DIR"], f"checkpoint-{iteration - 1}-{epochs_per_iteration:02d}.weights.keras")

        # Save the final model
        save_dir = config["FINAL_SAVE_DIR"]
        model.save(f"{save_dir}/final.h5")

    # wandb.finish()


if __name__ == "__main__":
    main()
