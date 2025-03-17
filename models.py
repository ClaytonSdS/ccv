try:
    from .losses import BinaryCrossEntropy
    from .losses import MeanSquaredError
    from .losses import CategoricalCrossEntropy

except ImportError:
    from losses import BinaryCrossEntropy
    from losses import MeanSquaredError
    from losses import CategoricalCrossEntropy

import jax.numpy as jnp
import jax
import tensorflow as tf
import numpy as np

def clip_gradients(gradients, clip_value=1.0):
    return jnp.clip(gradients, -clip_value, clip_value)

__all__ = ['Sequential']

class Sequential:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    # function to call the constructor method for each layer
    def build_layers(self, data_shape):
        batch_size = data_shape[0]
        primary_layer = self.layers[0]
        last_layer = self.layers[-1]
        last_function = last_layer.activation_function

        primary_shape= {'Dense': max(data_shape),
                        'Convolution':data_shape,
                        'BatchNorm':data_shape
                        # add other types of layers

        }
        input_shape = primary_shape[primary_layer.tag]

        pos = 1
        for layer in self.layers:
            out_tmp = layer.constructor(input_shape, batch_size)
            layer.pos_in_model = pos
            pos += 1

            print(f"layer: {layer.tag} input: {input_shape}: output: {out_tmp}")
            input_shape = layer.constructor(input_shape, batch_size)

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


    def fit(self, X_train:np.ndarray, Y_train:np.ndarray, epochs:int, batch_size:int=1):
        """
        Function that train the model using the provided data.

        Parameters:
        X_train (np.ndarray): Input data, expected shape (data, input_height, input_width, input_channels).
        Y_train (np.ndarray): Labels, expected shape (data, features, 1) with one-hot encoding.
        epochs (int): The number of times to iterate over the entire training dataset.
        batch_size (int, optional): The number of samples per gradient update. Default is 1.

        Returns:
        None: This function trains the model adjusting its parameters along the epochs
        """
        
        

        # Inicializando as variáveis para armazenar as losses e accuracies das épocas
        self.epoch_losses = []  # Para armazenar a loss de cada época
        self.epoch_accuracies = []  # Para armazenar a accuracy de cada época

        nan_break = False

        # looking for the last layer and passing useful parameters to it
        last_layer = self.layers[-1]
        lastActivation = last_layer.activation_function
        last_layer.model_loss = self.model_loss
        
        # setting x,y train as a tensorflow data type
        x_train = tf.data.Dataset.from_tensor_slices(X_train)
        y_train = tf.data.Dataset.from_tensor_slices(Y_train)

        dataset = tf.data.Dataset.zip((x_train, y_train))       # zipping the data in (x,y)
        dataset = dataset.shuffle(len(X_train))                 # shuffling the data
        dataset = dataset.batch(batch_size)                     # applying batch to it

        for first_sample in dataset.take(1):  # Taking the first sample to get batch, in_channels and input_size
            x, y = first_sample
            sample = list(x.numpy().shape)

            # len == 4, the network need to perform as convolutional
            if len(sample) == 4:
                # expected the shape be equal to (batch, in_channels, in_height, in_width)
                batch = sample[0]                       # get the first batch_size of the samples
                sample = sample[1:]                     # reshaping to have the format (in_channels, in_height, in_width)
                in_channels = min(sample)               # setting the input_channels to be the min(sample)
                sample.remove(in_channels)              # removing the in_channels, resulting in the shape (in_height, in_width)
                in_height, in_width =  sample
                first_data_shape = (batch, in_height, in_width, in_channels)        # 1st data shape that will be passed along the layers to build its tensors with random numbers -> (weights, biases, kernels, etc.)
                
                print(f"Len(4) 1st shape é: {first_data_shape}") # in_height, in_width, in_channels

            # len == 3, the network need to perform as dense

        self.build_layers(first_data_shape)

        nan_break = None

        # epochs' loop
        for e in range(epochs):
            total_loss = 0
            total_accuracy = 0
            total_samples = 0

            try:
                if tf.math.is_nan(nan_break): break         # break condition if nan on the backward prop.
            except ValueError: pass
            
            # loop for along the batches
            for x, y in dataset:
                batch = x.numpy().shape[0]       # taking the batch_size for each iteration, solving mismatches occurencies
                X = x.numpy()         # converting to jax numpy array
                Y = y.numpy()         # converting to jax numpy array

                output = X
                # forward propagation
                for layer in self.layers:
                    output = layer.forward(output, batch)

                    nan_break = output
                    try:
                        if tf.math.is_nan(nan_break): break         # break condition if nan on the backward prop.
                    except ValueError: pass
                    
                # Calcular o erro (loss) do batch
                batch_error = self.loss(Y, output)  # Média do erro sobre o batch
                total_loss += batch_error * batch_size  # Acumula o erro ponderado pelo tamanho do batch

                predicted_class = tf.argmax(output, axis=1)  # Previsão: classe com maior probabilidade
                true_class = tf.argmax(Y, axis=1)  # Classe verdadeira (one-hot encoding)
                
                correct_predictions = tf.equal(predicted_class, true_class)  # Verifica a correspondência
                accuracy_batch = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))  # Média dos acertos no batch
                total_accuracy += accuracy_batch  # Acumula a precisão do batch
                total_samples += 1  # Conta o número de batches processados

                # First gradient of the (epoch, dataset)
                gradient = self.loss.derivative(Y, output, lastActivation=lastActivation)
                gradient_norm = tf.norm(gradient)

                clip_threshold = 1.0


                epsilon = 1e-9
                # backward propagation
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learning_rate)
                    gradient = tf.clip_by_norm(gradient, 2.0)
                    gradient_norm = tf.norm(gradient)

                    print(f"[{layer.tag}{layer.pos_in_model}] Grad:\n{gradient_norm}")

                    if gradient_norm > clip_threshold:
                        2
                        #gradient = tf.clip_by_norm(gradient, clip_threshold)

                    if layer == self.layers[-2]:
                        2
                        #print(f"[{layer.tag}{layer.pos_in_model}] Grad:\n{gradient[0]}")

                    nan_break = gradient
                    try:
                        if tf.math.is_nan(nan_break): break         # break condition if nan on the backward prop.
                    except ValueError: pass


            self.epoch_losses.append(batch_error)
                        
            # Calculando a média da loss e da accuracy para a época
            avg_loss = total_loss / total_samples
            avg_accuracy = total_accuracy / total_samples

            self.epoch_losses.append(avg_loss)  # Armazena a loss da época
            self.epoch_accuracies.append(avg_accuracy)  # Armazena a accuracy da época

            print(f"Epoch [{e + 1}/{epochs}]: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

    def evaluate(self, x_test, y_test):
        pass
        # return {"acertos": (correct), "errors": int(errors), "accuracy": accuracy}
        

    def predict(self, x_test, batch_size=32, *kwargs):
        data_shape = (batch_size, *x_test[0].shape)

        print(f"O shape inicial é: {data_shape}")
        self.build_layers(data_shape)


        x_test = jnp.array(x_test).astype("float32")
        predictions = []
        num_samples = len(x_test)

        batch_predictions = []
        
        output = x_test
        for layer in self.layers:
            output = layer.forward(output)
            
        batch_predictions.append(output)

        return batch_predictions
    

