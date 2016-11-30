HISTOGRAMA
mlp_1_layer(100), mlp_1_layer(300), mlp_1_layer(500), mlp_1_layer(700), mlp_1_layer(900)
default encoding
0.203, 0.18, 0.185, 0.203, 0.211

mlp_2_layer(100, 900), mlp_2_layer(900, 100), mlp_2_layer(300, 700), mlp_2_layer(300, 300), mlp_2_layer(500, 500)
default encoding
0.098, 0.119, 0.109, 0.115, 0.098

CANAIS SOMADOS
encoding com todos os pixels tendo seus canais somados e standartized
10% (3000 iterations)

B&W STANDARTIZED

[mlp_1_layer(300)]
23% (3000 iterations)
26% (10000 iterations)

models = [mlp_1_layer(100), mlp_1_layer(300), mlp_1_layer(500), mlp_1_layer(700), mlp_1_layer(900)]
24%, 22%, 22%, 26%, 23%

models = [mlp_3_layer(300, 700, 300)]
25% (2000 iterações)

models = [mlp_3_layer(700, 500, 300)]
27% (2000 iterações)

models = [mlp_3_layer(900, 700, 500)]
27% (max: 29) (2000 iterações)

models = [mlp_3_layer(300, 300, 300)]
27% (2000 iterações)

models = [mlp_3_layer(300, 500, 700)]
10% (2000 iterações)

models = [mlp_1_layer(30)]
encoding com todos os pixels standartized
25% (10000)

RAW STANDARTIZED

[mlp_1_layer(300)]
25% (3000 iterations)

models = [mlp_1_layer(30)]
Usando sigmoid nos layers internos
28% (10000)

models = [mlp_2_layer(30, 30)]
Usando sigmoid nos layers internos
31% (2000)

models = [mlp_2_layer(300, 100)]
Usando sigmoid nos layers internos
31% (2000)

models = [mlp_2_layer(300, 300)]
Usando sigmoid nos layers internos
33% (2000)

models = [mlp_2_layer(300, 500)]
Usando sigmoid nos layers internos
30% (2000)

models = [mlp_2_layer(300, 300)]
Usando elu/sigmoid
31% (2000)

models = [mlp_2_layer(1500, 1500)]
Usando sigmoid nos layers internos
10% (não terminei de rodar)

models = [mlp_1_layer(1500)]
Usando sigmoid nos layers internos
10% (não terminei de rodar)

models = [mlp_3_layer(300, 300, 300)]
Usando sigmoid nos layers internos
32% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando sigmoid/relu/sigmoid
30% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando relu/relu/sigmoid
35% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando elu/elu/sigmoid
38% (2000) 35% numa segunda tentativa

models = [mlp_3_layer(300, 300, 300)]
 Usando softplus/softplus/sigmoid
36% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando elu/elu/tanh
10% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando tanh/tanh/tanh
26% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando tanh/tanh/sigmoid
27% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando relu6/relu6/sigmoid
26% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando relu/relu/relu
10% (2000)

models = [mlp_3_layer(300, 300, 300)]
Usando relu6/relu6/relu6
10% (2000)

models = [mlp_3_layer(300, 500, 300)]
Usando sigmoid nos layers internos
34% (2000)

models = [mlp_3_layer(300, 700, 300)]
Usando sigmoid nos layers internos
31% (2000) mais chegou a 35% T.T

models = [mlp_3_layer(900, 700, 500)]
Usando elu/elu/sigmoid
10% (2000)

models = [mlp_3_layer(30, 300, 30)]
Usando elu/elu/sigmoid
35% (2000)

models = [mlp_3_layer(700, 700, 700)]
Usando elu/elu/sigmoid
10% (2000)

models = [mlp_3_layer(500, 500, 500)]
Usando elu/elu/sigmoid
26% (2000)

models = [mlp_3_layer(200, 200, 200)]
Usando elu/elu/sigmoid
37% (2000) 40% max

models = [mlp_3_layer(150, 150, 150)]
Usando elu/elu/sigmoid
36% (2000) 38% max

models = [mlp_3_layer(100, 100, 100)]
Usando elu/elu/sigmoid
36% (2000)

models = [mlp_3_layer(50, 50, 50)]
Usando elu/elu/sigmoid
30% (2000)

models = [mlp_3_layer(30, 30, 30)]
Usando elu/elu/sigmoid
32% (2000)

models = [mlp_3_layer(10, 10, 10)]
Usando elu/elu/sigmoid
28% (2000)

models = [mlp_3_layer(500, 700, 500)]
Usando elu/elu/sigmoid
10% (2000)

mlp_3_layer_act(tf.nn.elu, 200, tf.nn.elu, 200, tf.nn.sigmoid, 200)
38% (10000)

mlp_3_layer_act(tf.nn.elu, 30, tf.nn.elu, 300, tf.nn.sigmoid, 30)
35% (10000)

mlp_3_layer_act(tf.nn.sigmoid, 300, tf.nn.sigmoid, 500, tf.nn.sigmoid, 300)
33% (10000)

mlp_3_layer_act(tf.nn.softplus, 300, tf.nn.softplus, 300, tf.nn.sigmoid, 300)
39% (10000)

mlp_3_layer_act(tf.nn.relu, 300, tf.nn.relu, 300, tf.nn.sigmoid, 300)
40% (10000)

RAW NORMALIZADO

mlp_1_layer(300)
sigmoid
37%/38% (2000)
39% (10000)

mlp_1_layer(30)
sigmoid
35% (2000)

mlp_1_layer(900)
sigmoid
10% (2000)

mlp_1_layer(2000)
sigmoid
10% (2000)

mlp_2_layer(300, 300)
elu, sigmoid
31% (2000)

mlp_3_layer_act(tf.nn.relu, 300, tf.nn.relu, 300, tf.nn.sigmoid, 300),
10% (2000)

mlp_3_layer_act(tf.nn.sigmoid, 300, tf.nn.sigmoid, 500, tf.nn.sigmoid, 300)
36% (2000)

mlp_3_layer_act(tf.nn.elu, 30, tf.nn.elu, 300, tf.nn.sigmoid, 30)
36%/39% (2000)

mlp_2_layer(300, 300)
sigmoid sigmoid
38% (2000)

mlp_3_layer_act(tf.nn.elu, 30, tf.nn.elu, 30, tf.nn.sigmoid, 30)
36% (2000)

mlp_1_layer(elu, 300)
10% (2000)

mlp_1_layer(tanh, 300),
10% (2000)

POSITION NORMALIZED

mlp_1_layer(sigmoid, 300),
28% (2000)

mlp_1_layer(sigmoid, 300),
23% (2000)

mlp_3_layer(elu, 300, elu, 300, sigmoid, 300)
10% (2000)

mlp_3_layer(sigmoid, 300, sigmoid, 300, sigmoid, 300)
28% (2000)

mlp_3_layer(tf.nn.elu, 30, tf.nn.elu, 300, tf.nn.sigmoid, 30)
37%

SIFT (10 clusters, 10000 iterations)
mlp_1_layer(sigmoid, 5), # 22%
mlp_1_layer(elu, 5), # 24%
mlp_1_layer(relu, 5), # 22%
mlp_1_layer(tanh, 5), # 20%
mlp_1_layer(tf.nn.softplus, 5), # 24%
mlp_1_layer(sigmoid, 5), # 23%
mlp_1_layer(sigmoid, 10), # 24%
mlp_1_layer(sigmoid, 15), # 23%
mlp_1_layer(sigmoid, 20), # 23%
mlp_1_layer(sigmoid, 30), # 24%
mlp_1_layer(sigmoid, 100), # 20
mlp_2_layer(sigmoid, 5, sigmoid, 5), # 21%
mlp_2_layer(elu, 5, sigmoid, 5), # 25%
mlp_2_layer(sigmoid, 5, elu, 5), # 24%
mlp_2_layer(elu, 5, elu, 5), # 22%
mlp_3_layer(sigmoid, 5, sigmoid, 5, sigmoid, 5), # 21%
mlp_3_layer(sigmoid, 5, sigmoid, 5, elu, 5), # 24%
mlp_3_layer(sigmoid, 5, elu, 5, sigmoid, 5), # 22%
mlp_3_layer(sigmoid, 5, elu, 5, elu, 5), # 22%
mlp_3_layer(elu, 5, sigmoid, 5, sigmoid, 5), # 24%
mlp_3_layer(elu, 5, sigmoid, 5, elu, 5), # 22%
mlp_3_layer(elu, 5, elu, 5, sigmoid, 5), # 21%
mlp_3_layer(elu, 5, elu, 5, elu, 5), # 25%

SIFT (100 clusters, 10000 iterations)
mlp_1_layer(sigmoid, 5), # 24%
mlp_1_layer(elu, 5), # 25%
mlp_1_layer(relu, 5), # 24%
mlp_1_layer(tanh, 5), # 25%
mlp_1_layer(tf.nn.softplus, 5), # 26%

mlp_1_layer(sigmoid, 5), # 23%
mlp_1_layer(sigmoid, 10), # 24%
mlp_1_layer(sigmoid, 15), # 23%
mlp_1_layer(sigmoid, 20), # 23%
mlp_1_layer(sigmoid, 30), # 21%
mlp_1_layer(sigmoid, 100), # 21%

mlp_2_layer(sigmoid, 5, sigmoid, 5), # 24%
mlp_2_layer(elu, 5, sigmoid, 5), # 22%
mlp_2_layer(sigmoid, 5, elu, 5), # 24%
mlp_2_layer(elu, 5, elu, 5), # 24%
mlp_3_layer(sigmoid, 5, sigmoid, 5, sigmoid, 5), # 21%
mlp_3_layer(sigmoid, 5, sigmoid, 5, elu, 5), # 25%
mlp_3_layer(sigmoid, 5, elu, 5, sigmoid, 5), # 22%
mlp_3_layer(sigmoid, 5, elu, 5, elu, 5), # 25%
mlp_3_layer(elu, 5, sigmoid, 5, sigmoid, 5), # 20%
mlp_3_layer(elu, 5, sigmoid, 5, elu, 5), # 23%
mlp_3_layer(elu, 5, elu, 5, sigmoid, 5), # 24%
mlp_3_layer(elu, 5, elu, 5, elu, 5), # 24%,

HoG
mlp_1_layer(tf.nn.sigmoid, 5)
ecoding com pixels não normalizados
treinamento: 10000
42,4%

HoG
mlp_1_layer(tf.nn.sigmoid, 5)
ecoding com pixels normalizados
treinamento: 10000
8,9%

HoG
mlp_1_layer(elu, 5),
ecoding com pixels não normalizados
treinamento: 10000
40,2%

HoG
mlp_1_layer(relu, 5),
ecoding com pixels não normalizados
treinamento: 10000
37,6%

treinamento: 9000
41,6%

HoG
mlp_1_layer(tanh, 5),
ecoding com pixels não normalizados
treinamento: 10000
41,2%

treinamento: 8900
42,8%

treinamento: 5400
43,7%

HoG
mlp_1_layer(tf.nn.softplus, 5)
encoding com pixels não normalizados
treinamento: 7200
44,2%

HoG
{
    'func': mlp_1_layer,
    'args': [elu, 300],
    'title': 'mlp 1 layer com elu'
 },
 20000
 48%

HoG
{
 'func': mlp_3_layer,
 'args': [relu, 300, relu, 300, sigmoid, 300],
 'title': 'mlp 1 layer com elu'
},
10000
46%

HoG
{
    'func': mlp_1_layer,
    'args': [relu, 500],
    'title': 'mlp 1 layer com sigmoid'
},
10000
47%

mlp_1_layer(tf.nn.softplus, 5)
ecoding com pixels não normalizados
treinamento: 19100
44,4%

HoG
mlp_2_layer(softplus, 5, softplus, 10)
ecoding com pixels não normalizados
treinamento: 7200
44,6%

HoG
mlp_2_layer(softplus, 5, softplus, 10)
ecoding com pixels não normalizados
treinamento: 7600
44,9%

HoG
mlp_3_layer(softplus, 5, softplus, 10, softplus, 10)
ecoding com pixels não normalizados
treinamento: 10000
40,4%
