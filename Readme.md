# Learn to approximate SDP solutions with GNN

## Dataset
The dataset contains 100 training examples, where each example contains the following data:
- (a_k, b_k), k=1,..,N: N pairs of 3D unit vectors representing the directional measurments to be aligned
- C: a symmetric matrix of size n x n (n = 4*N+4), which is the objective fuction for the SDP min <C, X>
- X: a symmetric matrix of size n x n, which is the primal optimal solution
- S: a symmetric matrix of size n x n, which is the dual optimal solution

I provide this dataset in a `.mat` file, as the solutions are generated by the MOSEK solver in Matlab. Note that I also calculate Aty = C - S. Because C is known, learning either S or Aty is equivalent.

This [line](https://github.com/hankyang94/GNNSDP/blob/5b880ff0a0b0af2ccfd5451c28a129379c32b7db/dataset.py#L42) shows how to read the `.mat` file and generate a Pytorch Geometric Dataset.

## Basic approach
- We start with an input fully-connected graph, where each node i has a feature
[a_k || b_k] of dimension 6
Optionally, we could use a different input feature that comes from the cost matrix C, see [here](https://github.com/hankyang94/GNNSDP/blob/5b880ff0a0b0af2ccfd5451c28a129379c32b7db/dataset.py#L66).
We add a dummy node 0 with feature all ones because the size of X and S is 4(N+1), with (N+1) blocks.

- We perform multiple layers of message passing on this input graph, and we generate a graph where each node has a feature f_i with high dimension.

- Given f_i, we then learn four nonlinear functions parametrized by MLPs:
    - primal_node(f_i), which returns the diagonal blocks of X
    - primal_edge(f_i,f_j), which returns the off-diagonal blocks of X
    - dual_node(f_i), which returns the diagonal blocks of S
    - dual_edge(f_i,f_j), which returns the off-diagonal blocks of S

- We perform supervised learning, where the loss function is:

loss = primal_loss + dual_loss

primal_loss = || X - X_gt || / || X_gt ||

dual_loss = || S - S_gt || / || S_gt ||

where all the norms are Frobenious norms.

A sammple implementation is [here](https://github.com/hankyang94/GNNSDP/blob/87d9697a2aec8ba1be88fac0fd97810df7e3ca73/model.py#L9).

