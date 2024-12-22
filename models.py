from losses import BinaryCrossEntropy
from losses import MeanSquaredError
from losses import CategoricalCrossEntropy
import jax.numpy as jnp


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

        for layer in self.layers:
            out_tmp = layer.constructor(input_shape)

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


    def fit(self, X_train, Y_train, epochs):
        last_layer = self.layers[-1]
        lastActivation = last_layer.activation_function
        last_layer.model_loss = self.model_loss
        
        print(f"O shape inicial é: {X_train[0].shape}")
        self.build_layers(data_shape=X_train[0].shape)

        self.epoch_losses = []  # Lista para armazenar a loss média de cada época

        for e in range(epochs):
            error = 0 
            print(f"epoch: {e}")
            for X, Y in zip(X_train, Y_train):
                # Forward Pass
                output = X
                for layer in self.layers:
                    output = layer.forward(output)
                
                # Verificação da saída após a função de perda
                error += self.loss(Y, output)
                #print(f"Erro após a função de perda: {error}, epoch: {e}")
                
                # Backward Pass
                gradient = self.loss.derivative(Y, output, lastActivation=lastActivation)
                #print(f"layer{layer} Gradiente inicial: {gradient}")

                # Verificação se o gradiente contém NaN ou Inf
                if jnp.any(jnp.isnan(gradient)):
                    print(f"[{epochs}] Gradiente contém NaN antes da retropropagação!, na camada {layer}")
                    break

                if jnp.any(jnp.isinf(gradient)):
                    print(f"[{epochs}] Gradiente contém Inf antes da retropropagação!, na camada {layer}")
                    break

                for layer in reversed(self.layers):
                    #gradient = clip_gradients(gradient)
                    gradient = layer.backward(gradient, self.learning_rate)

            mean_loss = error / len(X_train)
            self.epoch_losses.append(mean_loss)
                    
                
        
        error /= len(X_train)
        print(f"Resultado final {error}")

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

    def predict(self, x_test, *kwargs):
        predictions = []
        for x in x_test:
            output = x
            # Passando pela rede
            for layer in self.layers:
                output = layer.forward(output)

            # Pegando o índice da classe com maior probabilidade
            value = output[jnp.argmax(output)][0]
            pred = jnp.where(output==value, 1, jnp.where(output!=value, 0, output))
            predictions.append(pred)

        return jnp.array(predictions)
    
    def predict_one(self, x):
        output = x
        # Passando pela rede
        for layer in self.layers:
            output = layer.forward(output)
        return output


