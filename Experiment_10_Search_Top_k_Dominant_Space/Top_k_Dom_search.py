import torch
import pyhessian
import time
import numpy as np
from typing import List

def search_top_k_dominant_space(eigenvalues: List[float], method: str='gap'):
    """
    Search the top k dominant space
    Args:
        eigenvalues (list): A list of eigenvalues.
        method (str): The method to use for searching. Options are 'gap' or 'threshold'.
    Returns:
        int: The calculated top_k value. Returns 0 or len(eigenvalues) for edge cases in 'gap' method.
    """
    # Sort eigenvalues in descending order
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)

    if not sorted_eigenvalues:
        raise ValueError("The eigenvalues list is empty!")

    if method == 'gap':
        # Calculate gaps between consecutive eigenvalues
        gaps = [sorted_eigenvalues[i] - sorted_eigenvalues[i + 1] for i in range(len(sorted_eigenvalues) - 1)]
        # Find the index of the maximum gap
        max_gap_index = gaps.index(max(gaps))
        # The top k is the index of the maximum gap + 1
        top_k = max_gap_index + 1

        return top_k, gaps
    elif method == 'log_gaps':
        # Calculate log gaps between consecutive eigenvalues
        log_gaps = [np.log(sorted_eigenvalues[i] / sorted_eigenvalues[i + 1]) for i in range(len(sorted_eigenvalues) - 1)]
        # Find the index of the maximum log gap
        max_log_gap_index = log_gaps.index(max(log_gaps))
        # The top k is the index of the maximum log gap + 1
        top_k = max_log_gap_index + 1

        return top_k, log_gaps

    elif method == 'threshold':
        # TODO: Implement threshold method
        gaps = [0.0]
        top_k = len(sorted_eigenvalues)
        return top_k, gaps  # 添加这行
    else:
        raise ValueError(f"Unknown method: {method}")
    
   



    
    
