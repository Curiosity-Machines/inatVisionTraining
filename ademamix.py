import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
import math

class AdEMAMix(Optimizer):
    """AdEMAMix Optimizer.

    This optimizer extends Adam by introducing a slow EMA component to better leverage historical gradients.

    Arguments:
        learning_rate: A Tensor or a floating point value. The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay rate for the first moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay rate for the second moment estimates.
        beta_3: A float value or a constant float tensor. The exponential decay rate for the slow EMA.
        alpha: A float value controlling the influence of the slow EMA.
        T_alpha_beta3: An optional integer. If set, alpha_t and beta3_t are adjusted over time.
        epsilon: A small constant for numerical stability.
        weight_decay: Weight decay coefficient.
        name: Optional name for the operations created when applying gradients. Defaults to "AdEMAMix".
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 beta_3=0.9999,
                 alpha=5.0,
                 T_alpha_beta3=None,
                 epsilon=1e-8,
                 weight_decay=1e-4,
                 name="AdEMAMix",
                 **kwargs):
        super(AdEMAMix, self).__init__(name, **kwargs)
        
        # Initialize hyperparameters
        self._set_hyper("learning_rate", learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.alpha = alpha
        self.T_alpha_beta3 = T_alpha_beta3
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def _create_slots(self, var_list):
        # Create slots for the first, second, and slow EMA
        for var in var_list:
            self.add_slot(var, "m")        # Fast EMA (first moment)
            self.add_slot(var, "v")        # Second moment
            self.add_slot(var, "m_slow")   # Slow EMA

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Apply gradients to variables for dense tensors
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        
        # Retrieve the optimizer slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        m_slow = self.get_slot(var, "m_slow")
        
        # Current iteration step
        step = tf.cast(self.iterations + 1, var_dtype)
        lr_t = coefficients["lr_t"]
        beta1 = coefficients["beta1_t"]
        beta2 = coefficients["beta2_t"]
        epsilon = coefficients["epsilon"]
        weight_decay = self.weight_decay

        # Bias corrections
        bias_correction1 = 1.0 - tf.pow(beta1, step)
        bias_correction2 = 1.0 - tf.pow(beta2, step)

        # Compute alpha_t and beta3_t if T_alpha_beta3 is set
        if self.T_alpha_beta3 is not None:
            alpha_t = tf.minimum(step * self.alpha / self.T_alpha_beta3, self.alpha)
            log_beta1 = tf.math.log(beta1)
            log_beta3 = tf.math.log(self.beta_3)
            numerator = log_beta1 * log_beta3
            denominator = ((1.0 - step / self.T_alpha_beta3) * log_beta3 +
                           (step / self.T_alpha_beta3) * log_beta1)
            beta3_t = tf.minimum(tf.exp(numerator / denominator), self.beta_3)
        else:
            alpha_t = self.alpha
            beta3_t = self.beta_3

        # Update biased first moment estimate (m)
        m_t = m.assign(beta1 * m + (1.0 - beta1) * grad, use_locking=self._use_locking)
        
        # Update biased second moment estimate (v)
        v_t = v.assign(beta2 * v + (1.0 - beta2) * tf.square(grad), use_locking=self._use_locking)
        
        # Update slow EMA (m_slow)
        m_slow_t = m_slow.assign(beta3_t * m_slow + (1.0 - beta3_t) * grad, use_locking=self._use_locking)
        
        # Compute the denominator
        denom = (tf.sqrt(v_t) / tf.sqrt(bias_correction2)) + epsilon
        
        # Compute step size
        step_size = lr_t / bias_correction1

        # Apply weight decay if specified
        if weight_decay != 0.0:
            var.assign_sub(lr_t * weight_decay * var, use_locking=self._use_locking)
        
        # Update variable: var = var - step_size * (m + alpha_t * m_slow) / denom
        var_update = var.assign_sub(step_size * (m_t + alpha_t * m_slow_t) / denom,
                                   use_locking=self._use_locking)
        
        return tf.group(var_update, m_t, v_t, m_slow_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Apply gradients to variables for sparse tensors
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        
        # Retrieve the optimizer slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        m_slow = self.get_slot(var, "m_slow")
        
        # Current iteration step
        step = tf.cast(self.iterations + 1, var_dtype)
        lr_t = coefficients["lr_t"]
        beta1 = coefficients["beta1_t"]
        beta2 = coefficients["beta2_t"]
        epsilon = coefficients["epsilon"]
        weight_decay = self.weight_decay

        # Bias corrections
        bias_correction1 = 1.0 - tf.pow(beta1, step)
        bias_correction2 = 1.0 - tf.pow(beta2, step)

        # Compute alpha_t and beta3_t if T_alpha_beta3 is set
        if self.T_alpha_beta3 is not None:
            alpha_t = tf.minimum(step * self.alpha / self.T_alpha_beta3, self.alpha)
            log_beta1 = tf.math.log(beta1)
            log_beta3 = tf.math.log(self.beta_3)
            numerator = log_beta1 * log_beta3
            denominator = ((1.0 - step / self.T_alpha_beta3) * log_beta3 +
                           (step / self.T_alpha_beta3) * log_beta1)
            beta3_t = tf.minimum(tf.exp(numerator / denominator), self.beta_3)
        else:
            alpha_t = self.alpha
            beta3_t = self.beta_3

        # Update biased first moment estimate (m)
        m_scaled_g_values = (1.0 - beta1) * grad
        m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m.assign(beta1 * m, use_locking=self._use_locking)
        m.assign_add(m_t, use_locking=self._use_locking)
        
        # Update biased second moment estimate (v)
        v_scaled_g_values = (1.0 - beta2) * tf.square(grad)
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        v.assign(beta2 * v, use_locking=self._use_locking)
        v.assign_add(v_t, use_locking=self._use_locking)
        
        # Update slow EMA (m_slow)
        m_slow_scaled_g_values = (1.0 - beta3_t) * grad
        m_slow_t = self._resource_scatter_add(m_slow, indices, m_slow_scaled_g_values)
        m_slow.assign(beta3_t * m_slow, use_locking=self._use_locking)
        m_slow.assign_add(m_slow_t, use_locking=self._use_locking)
        
        # Compute the denominator
        denom = (tf.sqrt(v) / tf.sqrt(bias_correction2)) + epsilon
        
        # Compute step size
        step_size = lr_t / bias_correction1

        # Apply weight decay if specified
        if weight_decay != 0.0:
            var.assign_sub(lr_t * weight_decay * var, use_locking=self._use_locking)
        
        # Update variable: var = var - step_size * (m + alpha_t * m_slow) / denom
        var_update = var.assign_sub(step_size * (m + alpha_t * m_slow) / denom,
                                   use_locking=self._use_locking)
        
        return tf.group(var_update, m, v, m_slow)

    def get_config(self):
        # Return the configuration of the optimizer for serialization
        config = super(AdEMAMix, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "alpha": self.alpha,
            "T_alpha_beta3": self.T_alpha_beta3,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
        })
        return config

