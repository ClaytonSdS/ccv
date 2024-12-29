#%%
import jax.numpy as jnp


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

    @staticmethod
    def evaluate(y_true, y_pred):
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
        loss = -jnp.mean(y_true * jnp.log(y_pred), axis=0)
        print(f"Loss: {loss}")
        return loss

    @staticmethod
    def derivative_off(y_true, y_pred, **kwargs):
        # A derivada da Categorical Cross-Entropy é dada por:
        epsilon = 1e-7
        y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)  # Para evitar divisões por 0
        grad = - y_true / y_pred
        return grad

    @staticmethod
    def derivative(y_true, y_pred, **kwargs):
        if kwargs["lastActivation"] == "softmax":
            backward = y_pred - y_true
            print(f"[Loss-backward] {backward.shape}")

            if jnp.any(jnp.isnan(backward)):
                print(f"[Loss-backward]  NAN WARNING {jnp.sum(backward)}")
                return "nan"

            else:
                return backward
        
        else:
            epsilon = 1e-7
            y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
            grad = - y_true/y_pred
            return grad
