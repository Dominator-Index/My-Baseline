import numpy as np
from functools import wraps
# Binary Search for initial k_0
def Binary_Search(previous, current, tau, k_0):
    # tau = 0.95
    # Lower and upper bounds for k
    left, right = 0, k_0 - 1
    
    while left < right:
        mid = left + (right - left + 1) // 2  # 取较大中点，防止死循环
        matrix = np.matmul(previous[:, :mid].T, current[:, :mid])
        _, sigma, _ = np.linalg.svd(matrix)
        
        # Check if all singular values are within [0, 1]
        assert np.all((sigma >= -0.001) & (sigma <= 1.001)), "Eigenvectors are not ortho-normalized!"
    
        if np.min(sigma) < tau:
            right = mid - 1
        else:
            left = mid  
    
    return left + 1

# Define a decorator
def optimize_k0(func):
    @wraps(func)
    def wrapper(recorded_steps_top_eigenvectors, tau=0.98, k_0=None):
        sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
            
        if len(sorted_steps) < 2:
            raise ValueError("At least two recorded steps are required for Top-K search.")
        
        if k_0 is None:
        # 选择 epoch_0 和 epoch_1
            epoch_0 = recorded_steps_top_eigenvectors[sorted_steps[0]]
            epoch_1 = recorded_steps_top_eigenvectors[sorted_steps[1]]
            k_0 = epoch_0.shape[1]  # 先给 k_0 赋值
        
            # 计算初始 k_0
            optimized_k0 = Binary_Search(epoch_0, epoch_1, tau, k_0)
        else:
            optimized_k0 = k_0  # 直接使用提供的 k_0
            
        # 调用被装饰的 Top_Down 方法，并传入优化后的 k_0
        return func(recorded_steps_top_eigenvectors, tau, optimized_k0, sorted_steps)
    
    return wrapper

# Use the decorator
@optimize_k0

def Top_Down(recorded_steps_top_eigenvectors, tau=0.98, k_0=None, sorted_steps=None):
    if sorted_steps is None:  
        sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
        
    num_steps = len(sorted_steps) 
    # tau = 0.95 # Tolerance 
    k = int(k_0)  # Initialize k
    trajectory = {}
    trajectory[0] = k_0
    trajectory[1] = k_0 
    
    for i in range(2, num_steps):
        current_step = sorted_steps[i]
        current = recorded_steps_top_eigenvectors[current_step][:, :k]   # Current top k eigenvectors
        
        previous_step = sorted_steps[i - 1]
        previous = recorded_steps_top_eigenvectors[previous_step][:,:k]  # Previous top k eigenvectors
        
        if trajectory[i-1] == 0:
            trajectory[i] = 0
            continue
        
        for d in range(trajectory[i-1]):
            trajectory[i] = 0
            current_subspace = current[:, :trajectory[i-1]-d]  # Current subspace, top k-d eigenvectors
            previous_subspace = previous[:, :trajectory[i-1]-d]  # Previous subspace, top k-d eigenvectors
                
            # Compute the similarity between the current and previous subspaces
            matrix = np.matmul(current_subspace.T, previous_subspace)
            _, sigma, _ = np.linalg.svd(matrix)
                
            # Check if all singular values are within [0, 1]
            assert np.all((sigma >= -0.001) & (sigma <= 1.001)), "Eigenvectors are not ortho-normalized!"
                
            min_sigma = np.min(sigma)
                
            # If the minimal singular value is less then tau, then we reduce the dimensionality by d (one reduces dimension from top k to top k-d)
                
            # If the minimal singular value is greater than tau, which means the two subspaces are aligned, then we stop the loop and return the current k.
            if min_sigma >= tau:
                trajectory[i] = trajectory[i-1] - d
                break
            
    return trajectory

# This method is deprecated currently
def Bottom_Up(recorded_steps_top_eigenvectors):
    
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())  # Epoch
    num_steps = len(sorted_steps)
    
    if num_steps < 2:
        raise ValueError("At least two recorded steps are required for Top-K search.")
    
    # Initialize k
    k = recorded_steps_top_eigenvectors[sorted_steps[0]].shape[1]  # Number of columns
    # k = k_0
    tau = 0.98 # Tolerance
    
    for i in range(num_steps):
        current_step = sorted_steps[i]
        current = recorded_steps_top_eigenvectors[current_step][:, :k]   # Current top k eigenvectors
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous = recorded_steps_top_eigenvectors[previous_step][:,:k]  # Previous top k eigenvectors
            
            for d in range(k):
                
                current_subspace = current[:, :d+1]  # Current subspace
                previous_subspace = previous[:, :d+1]  # Previous subspace
                
                # Compute the similarity between the current and previous subspaces
                matrix = np.matmul(current_subspace.T, previous_subspace)
                _, sigma, _ = np.linalg.svd(matrix)
                
                # Check if all singular values are within [0, 1]
                assert np.all((sigma >= -0.001) & (sigma <= 1.001)), "Eigenvectors are not ortho-normalized!"
                
                # If the minimal singular value is less then tau, then we know that the two subspaces are not aligned, so we stop the loop and return the current k.
                min_sigma = np.min(sigma)
                
                if min_sigma < tau:
                    k = d
                    break

    return k

