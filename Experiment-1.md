算法描述
设在第 t 时刻，计算得到 n 维 Hessian 矩阵 H_t，其按特征值大小（绝对值或代价敏感顺序）降序排列的正交归一化特征向量矩阵为
$$\mathbf{V}_t = [\mathbf{v}_{t,1}, \mathbf{v}_{t,2}, \ldots, \mathbf{v}_{t,n}] \in \mathbb{R}^{n \times n}.$$
记 $\mathbf{V}_t^{(k)} = [\mathbf{v}_{t,1}, \ldots, \mathbf{v}_{t,k}]$ 为其前 k 个特征向量构成的子空间基。
给定相似度阈值 $\tau \in (0, 1]$，算法目标是在时间序列 $\{\mathbf{H}_1, \mathbf{H}_2, \ldots, \mathbf{H}_T\}$ 上，寻找最大的 k，使得对所有相邻时刻 t 与 t + 1，子空间 $\operatorname{span}(\mathbf{V}_t^{(k)})$ 与 $\operatorname{span}(\mathbf{V}_{t+1}^{(k)})$ 的最小奇异值均不低于 $\tau$。
具体步骤：
1. 初始化
- 设 $k_{\text{max}} = n$，将搜索起点 $k \leftarrow k_{\text{max}}$。
2. 对子空间相似度的度量
- 对任意给定 k，定义子空间相似度
$$
\sigma_{\min}^{(t)}(k) = \min\bigl(\text{singular\_values}((\mathbf{V}_t^{(k)})^\top \mathbf{V}_{t+1}^{(k)})\bigr)
$$

其中 $\sigma_{\text{min}}^{(t)}(k) \in [0, 1]$。
3. 逐步验证相似性
- 对所有 $t = 1, 2, \ldots, T - 1$：
a. 计算矩阵 $\mathbf{M}_t = (\mathbf{V}_t^{(k)})^\top \mathbf{V}_{t+1}^{(k)}$。
b. 对 $\mathbf{M}_t$ 做 SVD：
$$\mathbf{M}_t = \mathbf{U}_t \mathbf{\Sigma}_t \mathbf{W}_t^\top,$$
其中 $\mathbf{\Sigma}_t = \text{diag}(\sigma_1^{(t)}, \ldots, \sigma_k^{(t)})$，记 $\sigma_{\text{min}}^{(t)} = \sigma_k^{(t)}$。
c. 若对任意 t 有 $\sigma_{\text{min}}^{(t)} < \tau$，则认为当前 k 不满足要求，终止对该 k 的验证。

4. 调整 k
- 若所有 $\sigma_{\text{min}}^{(t)} \geq \tau$，则当前 k 合格；算法结束，输出该最大 k。
- 否则，将 $k \leftarrow k - 1$，返回步骤 3，直到 $k = 0$ 停止（若 $k = 0$ 仍不满足，则无任何非零维度子空间保持相似性）。

5. (可选) 二分加速
- 如需加速，可将上述线性搜索改为二分搜索：维护区间 $[\ell, u]$，初始 $[\ell = 0, u = n]$，反复取中点 $k = \lfloor (\ell + u) / 2 \rfloor$，测试相似性，若满足则 $\ell = k$，否则 $u = k$，直至收敛。


# Algorithm Description

Let $ \mathbf{H}_t \in \mathbb{R}^{n \times n} $ be the Hessian matrix computed at time step $ t $. Denote by  
$$
\mathbf{V}_t = [\mathbf{v}_{t,1}, \mathbf{v}_{t,2}, \ldots, \mathbf{v}_{t,n}] \in \mathbb{R}^{n \times n}
$$  
the orthonormal eigenvector matrix of $ \mathbf{H}_t $, whose columns are ordered according to eigenvalue magnitude (absolute value or cost-sensitive order) in descending order. Let  
$$
\mathbf{V}_t^{(k)} = [\mathbf{v}_{t,1}, \ldots, \mathbf{v}_{t,k}]
$$  
be the basis of the subspace spanned by the top $ k $ eigenvectors.

Given a similarity threshold $ \tau \in (0, 1] $, the goal of the algorithm is to find the largest $ k $ such that for every pair of consecutive time steps $ t $ and $ t+1 $ in the sequence $ \{\mathbf{H}_1, \mathbf{H}_2, \ldots, \mathbf{H}_T\} $, the minimal singular value between the subspaces $ \operatorname{span}(\mathbf{V}_t^{(k)}) $ and $ \operatorname{span}(\mathbf{V}_{t+1}^{(k)}) $ is at least $ \tau $.

1. **Initialization**  
   Set $ k_{\max} = n $, and initialize the search point $ k \leftarrow k_{\max} $.

2. **Similarity Metric for Subspaces**  
   For any given $ k $, define the subspace similarity at time $ t $ as  
   $$
   \sigma_{\min}^{(t)}(k) = \min\bigl(\text{singular\_values}((\mathbf{V}_t^{(k)})^\top \mathbf{V}_{t+1}^{(k)})\bigr),
   $$
   where $ \sigma_{\min}^{(t)}(k) \in [0,1] $.

3. **Iterative Similarity Verification**  
   For each $ t = 1, 2, \ldots, T-1 $:  
   a. Compute  
   $$
   \mathbf{M}_t = (\mathbf{V}_t^{(k)})^\top \mathbf{V}_{t+1}^{(k)}.
   $$  
   b. Perform singular value decomposition (SVD)  
   $$
   \mathbf{M}_t = \mathbf{U}_t \mathbf{\Sigma}_t \mathbf{W}_t^\top,
   $$
   where  
   $$
   \mathbf{\Sigma}_t = \operatorname{diag}(\sigma_1^{(t)}, \ldots, \sigma_k^{(t)}),
   $$
   and denote $ \sigma_{\min}^{(t)} = \sigma_k^{(t)} $ as the smallest singular value.  
   c. If for any $ t $, $ \sigma_{\min}^{(t)} < \tau $, then the current $ k $ fails the similarity criterion and the verification for this $ k $ terminates.

4. **Adjusting $ k $**  
   If all $ \sigma_{\min}^{(t)} \geq \tau $ for $ t = 1, \ldots, T-1 $, then the current $ k $ is valid; the algorithm stops and outputs this maximal $ k $.  
   Otherwise, update $ k \leftarrow k - 1 $ and return to Step 3. If $ k = 0 $ and the condition is still not met, then no nonzero-dimensional subspace maintains the required similarity.

5. **(Optional) Binary Search Acceleration**  
   To accelerate, replace the linear search with a binary search over the interval $[ \ell, u ]$ initialized as $ \ell = 0 $ and $ u = n $.  
   Iteratively select the midpoint  
   $$
   k = \left\lfloor \frac{\ell + u}{2} \right\rfloor,
   $$
   test the similarity condition: if satisfied, set $ \ell = k $; otherwise, $ u = k $. Continue until convergence.


This experiment aims to study the temporal persistence of the principal eigenvector direction of the Hessian during model training. Specifically, at each time step $t$, we extract the principal eigenvector of the Hessian matrix, denoted as $\mathbf{v}_{1}^{(t)}$. 

At the subsequent time step $t + 1$, we construct a weighted curvature matrix using the top $K$ eigenvalues $\{\lambda_{i}^{(t+1)}\}_{i=1}^{K}$ and their corresponding orthonormal eigenvectors $\{\mathbf{v}_{i}^{(t+1)}\}_{i=1}^{K}$ of the Hessian at time $t+1$:
$$
\mathbf{P}_{K}^{(t+1)} = \sum_{i=1}^{K} \lambda_{i}^{(t+1)} \mathbf{v}_{i}^{(t+1)} (\mathbf{v}_{i}^{(t+1)})^{\top}.
$$

We then compute the response intensity of the principal eigenvector $\mathbf{v}_{1}^{(t)}$ at step $t$ projected onto this weighted subspace, defined by the normalized projection value:
$$
r^{(t)} = \frac{\left\|\mathbf{P}_{K}^{(t+1)} \mathbf{v}_{1}^{(t)}\right\|_{2}}{\left\|\mathbf{v}_{1}^{(t)}\right\|_{2}}.
$$

