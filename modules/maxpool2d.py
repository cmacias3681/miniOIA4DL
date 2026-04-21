from modules.layer import Layer
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None

    def forward(self, input, training=True):
        self.input = input
        B, C, H, W = input.shape
        K = self.kernel_size
        S = self.stride

        # Calcular dimensiones de salida
        out_h = (H - K) // S + 1
        out_w = (W - K) // S + 1

        # --- OPTIMIZACIÓN: VECTORIZACIÓN CON STRIDES ---
        # Creamos una 'vista' de la memoria que organiza los datos en (B, C, out_h, out_w, K, K)
        # Esto no copia datos, solo cambia cómo NumPy accede a ellos.
        shape = (B, C, out_h, out_w, K, K)
        strides = (
            input.strides[0], 
            input.strides[1], 
            input.strides[2] * S, 
            input.strides[3] * S, 
            input.strides[2], 
            input.strides[3]
        )
        
        view = np.lib.stride_tricks.as_strided(input, shape=shape, strides=strides)
        
        # Calculamos el máximo sobre las últimas dos dimensiones (K, K)
        output = np.max(view, axis=(4, 5))
        
        # Para el backward (solo si es necesario para el concurso)
        if training:
            # Guardamos una máscara de dónde están los máximos para el gradiente
            # Nota: Esta versión simplificada es mucho más rápida
            self.view = view
            self.output = output
            
        return output

    def backward(self, grad_output, learning_rate=None):
        # Versión vectorizada del backward
        B, C, H, W = self.input.shape
        K, S = self.kernel_size, self.stride
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        
        grad_input = np.zeros_like(self.input)
        
        # Expandimos grad_output para que coincida con la vista de las ventanas
        # y aplicamos el gradiente solo donde el valor era igual al máximo
        for i in range(K):
            for j in range(K):
                # Comparamos la ventana con el valor máximo y propagamos gradiente
                mask = (self.input[:, :, i:i+out_h*S:S, j:j+out_w*S:S] == self.output)
                grad_input[:, :, i:i+out_h*S:S, j:j+out_w*S:S] += mask * grad_output
                
        return grad_input