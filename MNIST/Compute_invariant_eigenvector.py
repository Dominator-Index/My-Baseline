import numpy as np
from config import config
import wandb


def compute_invariant_matrix(W1, W2):
    """
    Computes the Kronecker product of the right singular vector of W1
    corresponding to the largest singular value and the right singular vector
    of W2.T corresponding to the largest singular value.

    Args:
        W1 (np.ndarray): The first matrix.
        W2 (np.ndarray): The second matrix.

    Returns:
        np.ndarray: The Kronecker product of the two singular vectors.
    """
    U_w1, sigma_w1, V_w1 = np.linalg.svd(W1)
    U_w2, sigma_w2, V_w2 = np.linalg.svd(W2.T)

    # The right singular vectors of W1 are the columns of V_w1.
    # The one corresponding to the largest singular value is the first column.
    right_singular_vector_w1 = V_w1[:, 0]


    # The one corresponding to the largest singular value is the first column.
    right_singular_vector_w2_t = V_w2[:, 0]

  
    
   
    return right_singular_vector_w1, right_singular_vector_w2_t




