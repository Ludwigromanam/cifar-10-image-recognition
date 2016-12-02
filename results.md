Esse arquivo contém os resultados que obtivemos em todos os testes que foram feitos.

Os melhores resultados estarão em destaque.

A não ser quando notado, os resultado foram obtidos com 10.000 iterações e todos os layers são softmax

# HISTOGRAMA
1 layer: (100 nós)
0.203%

1 layer: (300 nós)
0.18%

1 layer: (500 nós)
0.185%

1 layer: (700 nós)
0.203%

1 layer: (900 nós)
0.211%

2 layers: (100, 900)
0.098%

2 layers: (900, 100)
0.119%

2 layers: (300, 700)
0.109%

2 layers: (300, 300)
0.115%

2 layers: (500, 500)
0.098%

# CANAIS SOMADOS
## encoding com todos os pixels tendo seus canais somados e standartized
1 layer: (300)
10% (3000 iterations)

# Preto e Branco (standardize)

1 layer: (300)
23% (3000 iterações)
26%

1 layer: (100)
24%

1 layer: (300)
22%

1 layer: (500)
22%

1 layer: (700)
26%

1 layer: (900)
23%

1 layer: (30)
25%

3 layers: (300, 700, 300)
25% (2000 iterações)

3 layers: (700, 500, 300)
27% (2000 iterações)

3 layers: (900, 700, 500)
27% (2000 iterações)

3 layers: (300, 300, 300)
27% (2000 iterações)

3 layers: (300, 500, 700)
10% (2000 iterações)

# RAW (standardize)

1 layer: (300)
25% (3000 iterations)

1 layer: (30)
Usando sigmoid nos layers internos
28%

2 layers: (30, 30)
Usando sigmoid nos layers internos
31% (2000)

2 layers: (300, 100)
Usando sigmoid nos layers internos
31% (2000)

2 layers: (300, 300)
Usando sigmoid nos layers internos
33% (2000)

2 layers: (300, 500)
Usando sigmoid nos layers internos
30% (2000)

2 layers: (300, 300)
Usando elu/sigmoid
31% (2000)

2 layers: (1500, 1500)
Usando sigmoid nos layers internos
10%

1 layer: (1500)
Usando sigmoid nos layers internos
10%

3 layers: (300, 300, 300)
Usando sigmoid nos layers internos
32% (2000)

3 layers: (300, 300, 300)
Usando sigmoid/relu/sigmoid
30% (2000)

3 layers: (300, 300, 300)
Usando relu/relu/sigmoid
35% (2000)

3 layers: (300, 300, 300)
Usando elu/elu/sigmoid
38% (2000)

3 layers: (300, 300, 300)
Usando softplus/softplus/sigmoid
36% (2000)

3 layers: (300, 300, 300)
Usando elu/elu/tanh
10% (2000)

3 layers: (300, 300, 300)
Usando tanh/tanh/tanh
26% (2000)

3 layers: (300, 300, 300)
Usando tanh/tanh/sigmoid
27% (2000)

3 layers: (300, 300, 300)
Usando relu6/relu6/sigmoid
26% (2000)

3 layers: (300, 300, 300)
Usando relu/relu/relu
10% (2000)

3 layers: (300, 300, 300)
Usando relu6/relu6/relu6
10% (2000)

3 layers: (300, 500, 300)
Usando sigmoid nos layers internos
34% (2000)

3 layers: (300, 700, 300)
Usando sigmoid nos layers internos
31% (2000)

3 layers: (900, 700, 500)
Usando elu/elu/sigmoid
10% (2000)

3 layers: (30, 300, 30)
Usando elu/elu/sigmoid
35% (2000)

3 layers: (700, 700, 700)
Usando elu/elu/sigmoid
10% (2000)

3 layers: (500, 500, 500)
Usando elu/elu/sigmoid
26% (2000)

3 layers: (200, 200, 200)
Usando elu/elu/sigmoid
37% (2000) 40% max

3 layers: (150, 150, 150)
Usando elu/elu/sigmoid
36% (2000) 38% max

3 layers: (100, 100, 100)
Usando elu/elu/sigmoid
36% (2000)

3 layers: (50, 50, 50)
Usando elu/elu/sigmoid
30% (2000)

3 layers: (30, 30, 30)
Usando elu/elu/sigmoid
32% (2000)

3 layers: (10, 10, 10)
Usando elu/elu/sigmoid
28% (2000)

3 layers: (500, 700, 500)
Usando elu/elu/sigmoid
10% (2000)

3 layers: (200,200,200)
Usando elu/elu/sigmoid
38%

3 layers: (30,300,30)
Usando elu/elu/sigmoid
35%

3 layers: (300,500,300)
Usando sigmoid nos layers internos
33%

3 layers: (300,300,300)
Usando softplus/softplus/sigmoid
39%

3 layers: (300,300,300)
Usando relu/relu/sigmoid
40%

# RAW (normalized)

1 layer: (300)
Usando sigmoid nos layers internos
38% (2000)
39% (10000)

1 layer: (30)
Usando sigmoid nos layers internos
35% (2000)

1 layer: (900)
Usando sigmoid nos layers internos
10% (2000)

1 layer: (2000)
Usando sigmoid nos layers internos
10% (2000)

2 layers: (300, 300)
Usando elu, sigmoid nos layers internos
31% (2000)

3 layers: (300,300,300)
Usando relu/relu/sigmoid
10% (2000)

3 layers: (300,500,300)
Usando sigmoid nos layers internos
36% (2000)

3 layers: (30,300,30)
Usando elu/elu/sigmoid
39% (2000)

2 layers: (300, 300)
Usando sigmoid/sigmoid
38% (2000)

3 layers: (elu, 30, elu, 30, sigmoid, 30)
36% (2000)

1 layer: (elu, 300)
10% (2000)

1 layer: (tanh, 300)
10% (2000)

# SIFT

## 10 clusters
1 layer: (sigmoid, 5)                          # 22%
1 layer: (elu, 5)                              # 24%
1 layer: (relu, 5)                             # 22%
1 layer: (tanh, 5)                             # 20%
1 layer: (softplus, 5)                         # 24%

1 layer: (sigmoid, 5)                          # 23%
1 layer: (sigmoid, 10)                         # 24%
1 layer: (sigmoid, 15)                         # 23%
1 layer: (sigmoid, 20)                         # 23%
1 layer: (sigmoid, 30)                         # 24%
1 layer: (sigmoid, 100)                        # 20

2 layers: (sigmoid, 5, sigmoid, 5)              # 21%
2 layers: (elu, 5, sigmoid, 5)                  # 25%
2 layers: (sigmoid, 5, elu, 5)                  # 24%
2 layers: (elu, 5, elu, 5)                      # 22%

3 layers: (sigmoid, 5, sigmoid, 5, sigmoid, 5), # 21%
3 layers: (sigmoid, 5, sigmoid, 5, elu, 5)      # 24%
3 layers: (sigmoid, 5, elu, 5, sigmoid, 5)      # 22%
3 layers: (sigmoid, 5, elu, 5, elu, 5)          # 22%
3 layers: (elu, 5, sigmoid, 5, sigmoid, 5)      # 24%
3 layers: (elu, 5, sigmoid, 5, elu, 5)          # 22%
3 layers: (elu, 5, elu, 5, sigmoid, 5)          # 21%
3 layers: (elu, 5, elu, 5, elu, 5)              # 25%

## 100 clusters
1 layer: (sigmoid, 5)                          # 24%
1 layer: (elu, 5)                              # 25%
1 layer: (relu, 5)                             # 24%
1 layer: (tanh, 5)                             # 25%
1 layer: (softplus, 5)                         # 26%

1 layer: (sigmoid, 5)                          # 23%
1 layer: (sigmoid, 10)                         # 24%
1 layer: (sigmoid, 15)                         # 23%
1 layer: (sigmoid, 20)                         # 23%
1 layer: (sigmoid, 30)                         # 21%
1 layer: (sigmoid, 100)                        # 21%

2 layers: (sigmoid, 5, sigmoid, 5)              # 24%
2 layers: (elu, 5, sigmoid, 5)                  # 22%
2 layers: (sigmoid, 5, elu, 5)                  # 24%
2 layers: (elu, 5, elu, 5)                      # 24%

3 layers: (sigmoid, 5, sigmoid, 5, sigmoid, 5)  # 21%
3 layers: (sigmoid, 5, sigmoid, 5, elu, 5)      # 25%
3 layers: (sigmoid, 5, elu, 5, sigmoid, 5)      # 22%
3 layers: (sigmoid, 5, elu, 5, elu, 5)          # 25%
3 layers: (elu, 5, sigmoid, 5, sigmoid, 5)      # 20%
3 layers: (elu, 5, sigmoid, 5, elu, 5)          # 23%
3 layers: (elu, 5, elu, 5, sigmoid, 5)          # 24%
3 layers: (elu, 5, elu, 5, elu, 5)              # 24%

# HoG

## encoding com pixels normalizados
1 layer: (sigmoid, 5)
8,9%

## encoding com pixels não normalizados

1 layer: (sigmoid, 5)
42,4%

1 layer: (elu, 5)
40,2%

1 layer: (relu, 5)
37,6%
41,6% (9000 iterações)

1 layer: (tanh, 5)
41,2%
42,8% (8900 iterações)
43,7% (5400 iterações)

1 layer: (softplus, 5)
44,2% (7200 iterações)

1 layer: (elu, 300)
48,89% (1600 iterações)
49,2%
50,4% (7500 iterações)
48% (20000 iterações)

3 layers: (relu, 300, relu, 300, sigmoid, 300)
46%

1 layer: (relu, 500)
47%

1 layer: (softplus, 5)
44,4% (19100 iterações)

2 layers: (softplus, 5, softplus, 10)
44,6% (7200 iterações)

2 layers: (softplus, 5, softplus, 10)
44,9% (7600 iterações)

3 layers: (softplus, 5, softplus, 10, softplus, 10)
40,4%

3 layers: (softplus, 5, softplus, 20, softplus, 10)
42,4%
