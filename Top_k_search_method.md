### Algorithm

We can utilize the concept of **principal cosines** to measure the alignment between two subspaces. The specific steps are as follows:

---

### Theoretical Background

Given two matrices composed of column vectors:  
$$
U = [u_1, u_2, \dots, u_n],\quad V = [v_1, v_2, \dots, v_n],
$$
where the columns of each matrix are orthonormal. For any $ s $ ($ 1 \le s \le n $), define  
$$
U_s = [u_1, u_2, \dots, u_s],\quad V_s = [v_1, v_2, \dots, v_s].
$$
The **principal cosines** between the two $ s $-dimensional subspaces spanned by $ U_s $ and $ V_s $ are precisely the singular values of the matrix  
$$
M_s = U_s^T V_s.
$$
Denote these $ s $ singular values as  
$$
\sigma_1^{(s)} \ge \sigma_2^{(s)} \ge \cdots \ge \sigma_s^{(s)} \ge 0.
$$
Intuitively, when these singular values are close to 1, it indicates strong alignment between the subspaces across all directions. Conversely, the presence of smaller singular values implies poorer alignment in at least one direction.

---

### Algorithm Design

We aim to find the largest s $ s $ such that:

1. The alignment between subspaces $ \text{span}(u_1, \dots, u_s) $ and $ \text{span}(v_1, \dots, v_s) $ is "maximized" (i.e., we set a tolerance $ \tau $, requiring that all principal cosines are no less than $ \tau $, or simply, the smallest principal cosine satisfies $ \sigma_s^{(s)} \ge \tau $)。
2. When adding an additional vector (i.e., considering $ s+1 $), there is a significant gap in alignment for the new dimension (i.e., $ \sigma_{s+1}^{(s+1)} < \tau $)。

#### Detailed Steps:

1. **Select Tolerance $ \tau $：**  
   For example, choose $ \tau = 0.99 $ or another value based on practical requirements. This value represents the minimum directional cosine (i.e., a very small angle) required for alignment in each dimension.

2. **Iteratively Compute Principal Cosines:**  
   For $ s = 1, 2, \dots, n $：
   - Compute the matrix $ M_s = U_s^T V_s $。
   - Calculate all singular values $ M_s $ , and denote the smallest singular value as
   $$
   \sigma_{\min}^{(s)} = \min \{ \sigma_1^{(s)}, \dots, \sigma_s^{(s)} \}.
   $$
   - Check alignment criteria: 
     If $ \sigma_{\min}^{(s)} \ge \tau $，the subspaces spanned by the first $ s $ vectors are considered aligned within tolerance  $ \tau $.

3. **Identify the Critical Point:**  
   Increment $ s $ until a value $ s+1 $ satisfies 
   $$
   \sigma_{\min}^{(s+1)} < \tau.
   $$
   This indicates that while the first $ s $ directions are aligned, adding the ($ s+1 $)-th direction results in at least one direction failing the alignment tolerance.

4. **Output Result $ s $：**  
   Output Result s $ s $ Output Result s

---

### Mathematical Formulation

---

Define  
$$
s^* = \max \{ s \in \{1,\dots,n\} \mid \sigma_{\min}^{(s)} \ge \tau \},
$$  
Mathematical Formulation
$$
\sigma_{\min}^{(s^*+1)} < \tau \quad \text{（if } s^* < n \text{）}.
$$ 
Then $ s^* $ is the maximum dimension satisfying the alignment criteria.

### Intuitive Interpretation

- **For $ s = 1 $:**  
  Only vectors  $ u_1 $ and $ v_1 $. If $ |u_1^T v_1| \ge \tau $（(i.e., the angle between them is very small), they are considered aligned.

- **Gradually Increasing $ s $:**
  For each $ s $, verify that the newly formed $ s $-dimensional subspace remains globally aligned (all directional cosines meet $ \tau $ ).

- **Critical Point:**  
  When adding the ($ s+1 $)-th vector causes the smallest principal cosine to drop below $\tau$, ($ s+1 $)-th direction introduces significant misalignment. Thus $ s^* $ is the answer.

---

### Summary
Implementation steps:  
1. **Set tolerance $ \tau $。**  
2. **For each $ s = 1,2,\dots,n $, compute the smallest singular value $ \sigma_{\min}^{(s)}$ of $ M_s = U_s^T V_s $.**  
3. **Select the largest $ s $ s.t. $ \sigma_{\min}^{(s)} \ge \tau $ and $ \sigma_{\min}^{(s+1)} < \tau $（if $ s < n $）。**

This method leverages principal cosines (corresponding to singular values) to quantify subspace overlap and uses a threshold ττ to detect the onset of misalignment at a specific dimension.

This approach is both intuitive and practical. Computationally, it can be implemented by performing SVD for each $ s $.



        
        
