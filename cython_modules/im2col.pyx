# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

def im2col_forward_cython(
    float[:, :, :, :] input,  # Memoryview 4D
    int K,
    int S,
    int P
):
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]

    cdef int H_out = (H + 2 * P - K) // S + 1
    cdef int W_out = (W + 2 * P - K) // S + 1

    # Creamos el output como un array de numpy pero lo manejamos como memoryview
    cols_np = np.zeros((B * H_out * W_out, C * K * K), dtype=np.float32)
    cdef float[:, :] cols = cols_np

    cdef int b, c, i, j, ki, kj
    cdef int row, col_idx
    cdef int in_i, in_j

    # Si P > 0, podrías considerar hacer un np.pad previo para eliminar el IF.
    # Aquí mantenemos la lógica pero optimizada por el memoryview.
    
    for b in prange(B, nogil=True): # Paralelización opcional
        for i in range(H_out):
            for j in range(W_out):
                row = b * (H_out * W_out) + i * W_out + j
                for c in range(C):
                    for ki in range(K):
                        in_i = i * S + ki - P
                        for kj in range(K):
                            in_j = j * S + kj - P
                            col_idx = c * K * K + ki * K + kj
                            
                            if 0 <= in_i < H and 0 <= in_j < W:
                                cols[row, col_idx] = input[b, c, in_i, in_j]
                            else:
                                cols[row, col_idx] = 0.0
    return cols_np