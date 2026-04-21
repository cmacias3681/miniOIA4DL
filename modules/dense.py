from modules.utils import *
from modules.layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, in_features, out_features, weight_init="he"):
        self.in_features = in_features
        self.out_features = out_features

        # Inicialización de pesos optimizada
        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            std = 1.0 / np.sqrt(in_features)

        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.biases = np.zeros(out_features, dtype=np.float32)
        self.input = None

    def forward(self, input, training=True):
        # Aseguramos que el input sea float32 para velocidad y estabilidad
        self.input = np.asanyarray(input, dtype=np.float32)
        
        # OPTIMIZACIÓN: Usar el operador @ (matmul) de NumPy en lugar de funciones manuales
        # Esto aprovecha librerías BLAS altamente optimizadas
        output = (self.input @ self.weights) + self.biases
        self.output = output
        return output

    def backward(self, grad_output, learning_rate):
        grad_output = np.asanyarray(grad_output, dtype=np.float32)
        
        # OPTIMIZACIÓN RADICAL: Eliminamos los 3 niveles de bucles for
        # 1. Gradiente respecto a los pesos: dW = Input^T @ Grad_Output
        grad_weights = self.input.T @ grad_output
        
        # 2. Gradiente respecto a los biases: Suma de grad_output por columnas
        grad_biases = np.sum(grad_output, axis=0)

        # 3. Gradiente respecto a la entrada: dX = Grad_Output @ Weights^T
        grad_input = grad_output @ self.weights.T
        
        # Actualización de parámetros vectorizada
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
    
    def get_weights(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_weights(self, weights):
        self.weights = weights['weights']
        self.biases = weights['biases']