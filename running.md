# Running instances

--------Running-------

- primal train N30-1000, GNN_LAYER = 4, residual = False, 2000 epoches
- primal train N30-1000, GNN_LAYER = 6, residual = True, 800 epoches
    - N30-1000-20220510-182111
- primal train N30-1000, GNN_LAYER = 15, residual = True, 800 epoches
    - N30-1000-20220510-182020


--------Done------

- dual model, SAGEConv, input_dim=6, 2000 epoches, fully-connected, mlp_layer=2 (0.0743)
log: 20220509-111854
model: dual_model_SAGE_4_64_64_1_1_2000_0.0_2.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, mlp_layer=2, log 20220508-180608 (0.0654)
model: primal_model_SAGE_4_64_64_1_1_1000_0.0_2.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- dual model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, mlp_layer=1, log 20220508-185448 (0.1376)
model: dual_model_SAGE_4_64_64_1_1_1000_0.0_1.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, mlp_layer=1, log 20220508-111000 (0.0795)
primal_model_SAGE_4_64_64_1_1_1000_0_1.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal model, SAGEConv, input_dim=6, 500 epoches, fully-connected (seems to be the best now, 0.1512 loss), log 20220507-140346, model primal_model_SAGE_4_64_64_4_1.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected (0.1116)
model primal_model_SAGE_4_64_64_1_1_1000.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected (0.1417)
model primal_model_SAGE_6_64_64_1_1_1000.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 6

- primal train small TransformerConv
20220510-114122

- primal train small SAGE Residual
20220510-120156

- primal train small batchnorm
20220510-134109

------- Bad --------
- primal model, GATConv, input_dim=6, 500 epoches, fully_connected (0.8408 loss)
- primal model, SAGEConv, input_dim=6, 500 epoches, star (0.3149)
- primal model, GATConv, input_dim=6, 500 epoches, star, relu_slope = 0 (0.6938 loss)
- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, dropout=0.2
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4
- primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, mlp_layer=1, dropout=0.2
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- primal factor

- dual factor