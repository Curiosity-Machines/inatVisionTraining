import tensorflow as tf
from tensorflow import keras
from dynamic_dropout import DynamicDropout

class ProgressiveTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, make_dataset_fn):
        super(ProgressiveTrainingCallback, self).__init__()
        self.config = config
        self.make_dataset_fn = make_dataset_fn
        self.switch_epoch = config.get("PROGRESSIVE_TRAINING_EPOCH", 20)
        self.total_epochs = config["NUM_EPOCHS"]
        self.current_phase = 1

    def on_epoch_end(self, epoch, logs=None):
        if self.current_phase == 1 and (epoch + 1) == self.switch_epoch:
            print(f"\nSwitching to Phase 2 at epoch {epoch + 1}")
            
            # Update Dropout Rate
            for layer in self.model.layers:
                if isinstance(layer, DynamicDropout):
                    print(f"Updating dropout rate from {layer.rate.numpy()} to {self.config['DROPOUT_PCT']}")
                    layer.set_rate(self.config['DROPOUT_PCT'])
            
            # Rebuild the training dataset with new image size and augmentation
            new_image_size = self.config.get("FINAL_IMAGE_SIZE", [768, 768])
            print(f"Rebuilding training dataset with image size {new_image_size} and augmentation enabled.")
            self.model.fit_generator = None  # Reset any generator state if needed

            # Note: Re-initializing the dataset within the callback is non-trivial.
            # Instead, we'll handle dataset switching outside the callback in the main training loop.
            
            self.current_phase = 2

