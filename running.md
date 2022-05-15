# Running instances

--------Running-------


- primal train N30-1000
    - 9 layers, 64 neurons
    - MLP 1 hidden layer

- primal factor train N30-1000
    - 15-64-2
    - 9-96-2
    - 15-256-3


- primal train N30-1000, GNN_LAYER = 6, residual = True, 800 epoches, resmode=2
    - N30-1000-20220510-182539
    - 200 epoches: train acc: 0.3658, test acc: 0.7006, 0.6991
    - 400 epoches: 

- primal factor train N30-1000, GNN_LAYER = 3, residual=True, 1000 epoches, resmode=1
    - N30-1000-20220512-120422




--------Done------

- **** primal model with factor, SAGE, 7-64-64, 1000 epoches, residual, (0.0642)
    - primal_model_small_SAGE_7_True_False_factor_True.pth
    - learn nothing: test acc 1.0
    - 3-64-64 1000 epoches 0.2160 - 1.0491
    - 1-64-64 600 epoches 0.1590 - 0.9947
    

- ***** shared train small, GNN_LAYER = 7, residual = True, 800 epoches (this one has scheduler)
    - small-20220510-225416
    - shared_model_small_SAGE_7_True.pth

- shared train small, GNN_LAYER = 7, GNN_DIM = 64, residual = True, 800 epoches (no scheduler)
    - shared_model_small_SAGE_64_7_True.pth
    Train primal acc: 0.2880
    Train dual acc: 0.3227.
    Test primal acc: 0.9369
    Test dual acc: 0.3428.

- ***** dual model, SAGEConv, input_dim=6, 2000 epoches, fully-connected, mlp_layer=2 (0.0743)
log: 20220509-111854
model: dual_model_SAGE_4_64_64_1_1_2000_0.0_2.pth
    - GNN_HIDDEN_DIM = 64
    - GNN_OUT_DIM = 64
    - GNN_LAYER = 4

- ***** primal model, SAGEConv, input_dim=6, 1000 epoches, fully-connected, mlp_layer=2, log 20220508-180608 (0.0654)
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

- primal train N30-1000, GNN_LAYER = 15, residual = True, 800 epoches
    - N30-1000-20220510-182646