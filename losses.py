#%%
import tensorflow as tf
import jax.numpy as jnp
import keras

class MeanSquaredError:
    def __new__(cls, y_true, y_pred):
        return cls.evaluate(y_true, y_pred)

    @staticmethod
    def evaluate(y_true, y_pred):
        return jnp.mean(jnp.power(y_true - y_pred, 2))

    @staticmethod
    def derivative(y_true, y_pred, **kwargs):
        return 2 * (y_pred - y_true) / jnp.size(y_true)
    
# ////////////////////////////////////////////////////

class BinaryCrossEntropy:
    def __new__(cls, y_true, y_pred):
        return cls.evaluate(y_true, y_pred)

    @staticmethod
    def evaluate(y_true, y_pred):
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        e1 = -y_true * jnp.log(y_pred)
        e2 = (1 - y_true) * jnp.log(1 - y_pred)
        e = jnp.mean(e1 - e2)
        return e

    @staticmethod
    def derivative(y_true, y_pred, **kwargs):
        grad = ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / jnp.size(y_true)
        return grad


class CategoricalCrossEntropy:
    def __new__(cls, y_true, y_pred):
        return cls.evaluate(y_true, y_pred)

    def evaluate(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        #print(f"y_pred: {y_pred.shape} | y_true: {y_true.shape}")
        loss_keras = keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size', axis=-1)
        loss_keras = float(loss_keras(y_true, y_pred))


        # nan check
        if tf.reduce_any(tf.math.is_nan(y_pred)).numpy():
            print(f"[Loss-forward] nan found in y_pred[{y_pred.shape}]")
        #
        #print(f"[CategoricalCrossEntropy] keras_loss: {loss_keras}")
        return loss_keras

    def derivative(y_true, y_pred, **kwargs):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            tape.watch(y_pred)  # Definindo y_pred como um tensor a ser observado
            loss_value = loss_fn(y_true, y_pred)  # Calcula a perda novamente dentro do tape

        gradient = tape.gradient(loss_value, y_pred)
        gradient = tf.reshape(gradient, [*gradient.shape,1])
        #print(f"Loss gradient: \n{gradient[0]}")
        return gradient

    
