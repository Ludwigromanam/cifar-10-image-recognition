import encoder_hog_scikit_image as encoder
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


# help function to sampling data
def get_sample(num_samples, X_data, y_data):
    positions = np.arange(len(y_data))
    np.random.shuffle(positions)

    X_sample = []
    y_sample = []

    for posi in positions[:num_samples]:
        X_sample.append(X_data[posi])
        y_sample.append(y_data[posi])

    return X_sample, y_sample


######################## creating the model architecture #######################################

INPUT_SIZE = 1024
SIGMOID = tf.nn.sigmoid
ELU = tf.nn.elu
RELU = tf.nn.relu
TANH = tf.nn.tanh
SOFTPLUS = tf.nn.softplus

def mlp_1_layer(l1_act, layer_size):

    # input placeholder
    # mudei pra 1024 pra caber o hog encode
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    # output placeholder
    y_ = tf.placeholder(tf.float32, [None, 10])


    # weights of the neurons in first layer
    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, layer_size], stddev=0.35))
    b1 = tf.Variable(tf.random_normal([layer_size], stddev=0.35))

    # weights of the neurons in second layer
    W2 = tf.Variable(tf.random_normal([layer_size,10], stddev=0.35))
    b2 = tf.Variable(tf.random_normal([10], stddev=0.35))

    # hidden_layer value
    hidden_layer = l1_act(tf.matmul(x, W1) + b1)

    # output of the network
    y_estimated = tf.nn.softmax(tf.matmul(hidden_layer, W2) + b2)

    return x, y_, y_estimated

def mlp_2_layer(l1_act, l1_size, l2_act, l2_size):

    # input placeholder
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])



    # output placeholder
    y_ = tf.placeholder(tf.float32, [None, 10])


    # weights of the neurons in first layer
    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, l1_size], stddev=0.35))
    b1 = tf.Variable(tf.random_normal([l1_size], stddev=0.35))

    # weights of the neurons in second layer
    W2 = tf.Variable(tf.random_normal([l1_size,l2_size], stddev=0.35))
    b2 = tf.Variable(tf.random_normal([l2_size], stddev=0.35))

    W3 = tf.Variable(tf.random_normal([l2_size,10], stddev=0.35))
    b3 = tf.Variable(tf.random_normal([10], stddev=0.35))

    hidden_layer = l1_act(tf.matmul(x, W1) + b1)

    hidden_layer2 = l2_act(tf.matmul(hidden_layer, W2) + b2)

    # output of the network
    y_estimated = tf.nn.softmax(tf.matmul(hidden_layer2, W3) + b3)

    return x, y_, y_estimated

def mlp_3_layer(l1_act, l1_size, l2_act, l2_size, l3_act, l3_size):

    # input placeholder

    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    # output placeholder
    y_ = tf.placeholder(tf.float32, [None, 10])


    # weights of the neurons in first layer

    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, l1_size], stddev=0.35))
    b1 = tf.Variable(tf.random_normal([l1_size], stddev=0.35))

    # weights of the neurons in second layer
    W2 = tf.Variable(tf.random_normal([l1_size,l2_size], stddev=0.35))
    b2 = tf.Variable(tf.random_normal([l2_size], stddev=0.35))

    W3 = tf.Variable(tf.random_normal([l2_size,l3_size], stddev=0.35))
    b3 = tf.Variable(tf.random_normal([l3_size], stddev=0.35))

    W4 = tf.Variable(tf.random_normal([l3_size,10], stddev=0.35))
    b4 = tf.Variable(tf.random_normal([10], stddev=0.35))

    hidden_layer = l1_act(tf.matmul(x, W1) + b1)

    hidden_layer2 = l2_act(tf.matmul(hidden_layer, W2) + b2)

    hidden_layer3 = l3_act(tf.matmul(hidden_layer2, W3) + b3)

    # output of the network
    y_estimated = tf.nn.softmax(tf.matmul(hidden_layer3, W4) + b4)

    return x, y_, y_estimated


X_train, y_train, X_validation, y_validation, X_test, y_test = encoder.encode()

models = [
    # {
    #     'func': mlp_3_layer,
    #     'args': [tanh, 300, tanh, 300, sigmoid, 300],
    #     'title': 'mlp 1 layer com elu'
    #  },
    {
        'func': mlp_1_layer,
        'args': [ELU, 600],
        'title': 'mlp 1 layer com elu 600 nos'
    },
    # {
    #     'func': mlp_1_layer,
    #     'args': [relu, 5],
    #     'title': 'mlp 1 layer com relu'
    #  },
    # {
    #     'func': mlp_1_layer,
    #     'args': [softplus, 5],
    #     'title': 'mlp 1 layer com softplus'
    #  },
    # {
    #     'func': mlp_1_layer,
    #     'args': [tanh, 5],
    #     'title': 'mlp 1 layer com tanh'
    # },
    # {
    #     'func': mlp_2_layer,
    #     'args': [softplus, 5, softplus, 10],
    #     'title': 'mlp 2 layer com softplus'
    #  },
    # {
    #     'func': mlp_3_layer,
    #     'args': [softplus, 5, softplus, 10, softplus, 10],
    #     'title': 'mlp 3 layer com softplus'
    #  },
    # {
    #     'func': mlp_2_layer,
    #     'args': [softplus, 5, softplus, 10],
    #     'title': 'mlp 2 layer com softplus'
    #  },
    # {
    #     'func': mlp_3_layer,
    #     'args': [softplus, 5, softplus, 10, softplus, 10],
    #     'title': 'mlp 3 layer com softplus'
    #  },
    # {
    #     'func': mlp_2_layer,
    #     'args': [SOFTPLUS, 5, SOFTPLUS, 10],
    #     'title': 'mlp 2 layer com softplus 4x4'
    #  },
    # {
    #     'func': mlp_3_layer,
    #     'args': [softplus, 5, softplus, 20, softplus, 10],
    #     'title': 'mlp 3 layer com softplus 20 hidden layer'
    #  },
    # {
    #     'func': mlp_2_layer,
    #     'args': [softplus, 5],
    #     'title': 'mlp 1 layer com softplus'
    #  },
    # {
    #     'func': mlp_1_layer,
    #     'args': [tanh, 5],
    #     'title': 'mlp 1 layer com tanh'
    # },
]

# for ret_net, title in models:
for model in models:
    net = model.get('func')
    title = model.get('title')
    x, y_, y_estimated = net(*model.get('args'))

    # function to measure the error
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_estimated), reduction_indices=[1]))


    # how to train the model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    # how to evaluate the model
    correct_prediction = tf.equal(tf.argmax(y_estimated,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    ######################## training the model #######################################

    # applying a value for each variable (in this case W and b)
    init = tf.initialize_all_variables()


    # a session is dependent of the enviroment where tensorflow is running
    sess = tf.Session()
    sess.run(init)

    num_batch_trainning = 500
    x_grafico = []
    y_grafico = []


    iteracoes = 60000

    melhores_acuracia = []
    for i in range(iteracoes):
        # randomizing positions
        X_sample, y_sample = get_sample(num_batch_trainning, X_train, y_train)

        # where the magic happening
        sess.run(train_step, feed_dict={x: X_sample, y_:  y_sample})

        # print the accuracy result
        if i % 100 == 0:
            acuracia_atual = (sess.run(accuracy, feed_dict={x: X_validation, y_: y_validation}))
            x_grafico.append(i)
            y_grafico.append(acuracia_atual)
            print i, ": ", acuracia_atual
            if acuracia_atual > 0.499:
                melhores_acuracia.append((i, acuracia_atual))

    print "\n\n\n"
    print "TEST RESULT: ", (sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))
    print "Melhores acuracias: {}".format(melhores_acuracia)
    plot_name = "../graficos/hog/{}-{}.png".format(title, iteracoes)
    plt.plot(x_grafico, y_grafico)
    plt.title(title)
    plt.xlabel("Numero de iteracoes")
    plt.ylabel("Acuracia")
    plt.savefig(plot_name)
    plt.show()
