models = [mlp_1_layer(100), mlp_1_layer(300), mlp_1_layer(500), mlp_1_layer(700), mlp_1_layer(900)]
default encoding
0.203, 0.18, 0.185, 0.203, 0.211

models = [mlp_2_layer(100, 900), mlp_2_layer(900, 100), mlp_2_layer(300, 700), mlp_2_layer(300, 300), mlp_2_layer(500, 500)]
default encoding
0.098, 0.119, 0.109, 0.115, 0.098

models = [mlp_1_layer(300)]
encoding com todos os pixels standartized
25% (3000 iterations)
encoding com todos os pixels tendo seus canais somados e standartized
10% (3000 iterations)
encoding com todos os pixels preto e branco e standartized
23% (3000 iterations)
26% (10000 iterations)

models = [mlp_1_layer(100), mlp_1_layer(300), mlp_1_layer(500), mlp_1_layer(700), mlp_1_layer(900)]
encoding com todos os pixels preto e branco e standartized
24%, 22%, 22%, 26%, 23%
