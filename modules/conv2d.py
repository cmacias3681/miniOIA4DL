from modules.layer import Layer
from modules.utils import *
from cython_modules.im2col import im2col_forward_cython

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Selección del modo de convolución
        if conv_algo == 0:
            self.mode = 'direct' 
        elif conv_algo == 1:
            self.mode = 'im2col'
        elif conv_algo == 2:
             self.mode = 'im2col_cython'
        else:
            raise ValueError(f"Algoritmo {conv_algo} no soportado")

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        # Inicialización de pesos
        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        
        self.biases = np.zeros(out_channels, dtype=np.float32)

        # Parámetros para optimización de caché (Blocking/Tiling)
        self.mc, self.nc, self.kc = 480, 3072, 384
        self.mr, self.nr = 32, 12

    def forward(self, input, training=True):
        self.input = input
        # Despacho según el modo seleccionado
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            # Implementación im2col básica (Python/NumPy)
            return self._forward_im2col_numpy(input)
        elif self.mode == 'im2col_cython':
            return self._forward_im2col_cython(input)
        else:
            raise ValueError("Modo de convolución no válido")

    def _forward_im2col_numpy(self, input):
        """Implementación Opción 1: im2col vectorizado con NumPy strides"""
        B, C, H, W = input.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        
        # Aplicar padding si es necesario
        input_p = np.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant') if P > 0 else input
        
        H_p, W_p = input_p.shape[2], input_p.shape[3]
        H_out = (H_p - K) // S + 1
        W_out = (W_p - K) // S + 1

        # Generar vista de columnas usando strides (evita bucles for)
        shape = (B, C, K, K, H_out, W_out)
        strides = (input_p.strides[0], input_p.strides[1], input_p.strides[2], input_p.strides[3], 
                   input_p.strides[2] * S, input_p.strides[3] * S)
        
        view = np.lib.stride_tricks.as_strided(input_p, shape=shape, strides=strides)
        col = view.transpose(0, 4, 5, 1, 2, 3).reshape(B * H_out * W_out, -1)

        # Multiplicación GEMM
        W_col = self.kernels.reshape(self.out_channels, -1)
        out = col @ W_col.T + self.biases

        return out.reshape(B, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2).astype(np.float32)
    
    def _forward_im2col_cython(self, input):
        B, C, H, W = input.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        col = im2col_forward_cython(input, K, S, P)
        
        W_col = self.kernels.reshape(self.out_channels, -1)

        out = col @ W_col.T
        out += self.biases

        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        out = out.reshape(B, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out.astype(np.float32)

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c, i*self.stride : i*self.stride+k_h, j*self.stride : j*self.stride+k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]
        return output

# --- IMPLEMENTACIÓN BACKWARD (VECTORIZADA) ---

    def backward(self, grad_output, learning_rate):
        B, C, H, W = self.input.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        OC = self.out_channels

        # 1. Gradiente de Biases: Sumar sobre batch y dimensiones espaciales
        grad_biases = np.sum(grad_output, axis=(0, 2, 3))

        # 2. Gradiente de Pesos (dW): dL/dY * Col^T
        # grad_output -> (OC, B * H_out * W_out)
        grad_flat = grad_output.transpose(1, 0, 2, 3).reshape(OC, -1)
        
        # Usamos la matriz 'col' guardada en el forward para evitar recalcular
        grad_weights = (grad_flat @ self.col).reshape(self.kernels.shape)

        # 3. Gradiente de Entrada (dX): W^T * dL/dY
        W_flat = self.kernels.reshape(OC, -1)
        dX_col = W_flat.T @ grad_flat
        
        # Transformar columnas de vuelta a imagen (col2im)
        dX = self._col2im(dX_col, B, C, H, W, K, S, P)

        # 4. Actualización de pesos (SGD)
        self.kernels -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return dX

    def _col2im(self, col, B, C, H, W, K, S, P):
        H_p, W_p = H + 2*P, W + 2*P
        img_p = np.zeros((B, C, H_p, W_p), dtype=np.float32)
        H_out, W_out = (H_p - K) // S + 1, (W_p - K) // S + 1
        
        col_reshaped = col.T.reshape(B, H_out, W_out, C, K, K)
        for i in range(H_out):
            for j in range(W_out):
                img_p[:, :, i*S:i*S+K, j*S:j*S+K] += col_reshaped[:, i, j, :, :, :]
        
        return img_p[:, :, P:H_p-P, P:W_p-P] if P > 0 else img_p

    # --- GEMM OPTIMIZADO (Boceto para uso de mc, nc, kc) ---
    def _gemm_optimized(self, A, B):
        """
        Ejemplo de cómo aplicarías el tiling que tienes en el __init__.
        A: (M, K), B: (K, N)
        """
        M, K_dim = A.shape
        N = B.shape[1]
        C = np.zeros((M, N), dtype=np.float32)

        for i in range(0, M, self.mc):
            i_lim = min(i + self.mc, M)
            for k in range(0, K_dim, self.kc):
                k_lim = min(k + self.kc, K_dim)
                # Tiling / Packing logic here...
                C[i:i_lim, :] += A[i:i_lim, k:k_lim] @ B[k:k_lim, :]
        return C