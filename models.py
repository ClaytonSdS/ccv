from losses import BinaryCrossEntropy
from losses import MeanSquaredError
from losses import CategoricalCrossEntropy
import jax.numpy as jnp
import jax


def clip_gradients(gradients, clip_value=1.0):
    return jnp.clip(gradients, -clip_value, clip_value)

__all__ = ['Sequential']

class Sequential:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    # function to call the constructor method for each layer
    def build_layers(self, data_shape):
        primary_layer = self.layers[0]
        last_layer = self.layers[-1]
        last_function = last_layer.activation_function

        primary_shape= {'Dense': max(data_shape),
                        'Convolution':data_shape,
                        # add other types of layers

        }
        input_shape = primary_shape[primary_layer.tag]

        pos = 1
        for layer in self.layers:
            out_tmp = layer.constructor(input_shape)
            layer.pos_in_model = pos
            pos += 1

            print(f"layer: {layer.tag} input: {input_shape}: output: {out_tmp}")
            input_shape = layer.constructor(input_shape)

        last_layer.islastlayer = True
        last_layer.lastActivation = last_function

            

    def compile(self, loss, optimizer=None):
        # defining the loss function
        loss_functions = {
            'mse': MeanSquaredError,
            'binary_crossentropy': BinaryCrossEntropy,
            'categorical_crossentropy': CategoricalCrossEntropy
        }
        self.loss = loss_functions[loss]
        self.model_loss = loss

        # defining the optimizer
        optimizers = {
            'learning_rate':0
        }


    def fit(self, X_train, Y_train, epochs, batch_size=1):
        last_layer = self.layers[-1]
        lastActivation = last_layer.activation_function
        last_layer.model_loss = self.model_loss

        data_shape = (batch_size, *X_train[0].shape)

        print(f"O shape inicial é: {data_shape}")
        self.build_layers(data_shape)

        self.epoch_losses = []  # Lista para armazenar a loss média de cada época

        nan_break = False

        for e in range(epochs):
            error = 0
            print(f"\nEpoch: {e}")

            if nan_break:
                break
            
            # Embaralhar os dados para garantir que o treinamento seja aleatório
            self.rng = jax.random.PRNGKey(11) 
            permutation = jax.random.permutation(self.rng, len(X_train))
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]
            
            # Loop por lotes
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                X_batch = X_train_shuffled[start:end]
                Y_batch = Y_train_shuffled[start:end]

                # Forward Pass para todo o lote
                output = X_batch
                for layer in self.layers:
                    layer.Y_BATCH = Y_batch
                    
                    output = layer.forward(output)

                # Calcular o erro para o lote
                batch_error = jnp.mean(self.loss(Y_batch, output))  # Média do erro sobre o lote
                error += batch_error

                
                # Backward Pass para todo o lote
                gradient = self.loss.derivative(Y_batch, output, lastActivation=lastActivation)

                if gradient == "nan":
                    nan_break = True
                    print(f"NAN ERROR in LOSS")
                    break
                
                epsilon = 1e-9
                # Retropropagação para todas as camadas
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)
                    gradient = jnp.clip(gradient, epsilon, 1/epsilon)

                    if gradient == "nan":
                        nan_break = True
                        print(f"NAN ERROR in {layer.tag}-{layer.pos_in_model} layer")
                        break


            mean_loss = error / (len(X_train) // batch_size)
            self.epoch_losses.append(mean_loss)
                        
                    
            
            error /= len(X_train)
            print(f"[{e+1}/{epochs}] loss: {error}")

    def evaluate(self, x_test, y_test):
        # Fazendo a predição para o conjunto de teste
        predictions = self.predict(x_test)
        
        # Pegando o índice da classe com maior probabilidade
        pred_classes = jnp.argmax(predictions, axis=-1)
        
        # Pegando o índice da classe verdadeira
        true_classes = jnp.argmax(y_test, axis=-1)
        
        # Comparando as classes previstas com as classes reais
        correct = jnp.sum(pred_classes == true_classes)
        total = len(y_test)
        
        # Calculando acertos e erros
        accuracy = correct / total
        errors = total - correct
        
        # Retornando o dicionário com acertos e erros
        return {"acertos": int(correct), "erros": int(errors), "accuracy": accuracy}

    def predict(self, x_test, batch_size=32, *kwargs):
        x_test = jnp.array(x_test).astype("float32")
        predictions = []
        num_samples = len(x_test)

        batch_predictions = []
        
        output = x_test
        for layer in self.layers:
            output = layer.forward(output)
            
        batch_predictions.append(output)

        return batch_predictions
    
    def predict_one(self, x):
        output = x
        # Passando pela rede
        for layer in self.layers:
            output = layer.forward(output)
        return output


