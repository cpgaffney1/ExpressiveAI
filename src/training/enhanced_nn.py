from src import util
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
import keras.layers as layers
import keras.regularizers as reg
from src.training import enhanced_nn_predict as predict
from keras.initializers import TruncatedNormal
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras.backend as K
import datetime
import tensorflow as tf

BATCH_SIZE = 512

difference_mat = np.zeros((BATCH_SIZE * 2 - 1, BATCH_SIZE * 2))
for i in range(difference_mat.shape[0]):
    for j in range(difference_mat.shape[1]):
        if i == j:
            difference_mat[i][j] = -1
        if i + 1 == j:
            difference_mat[i][j] = 1
print(difference_mat)

def custom_loss(y_true, y_pred):
    reg_weight = 0.001
    square_mat = K.dot(y_pred, K.transpose(y_pred))
    y_flat = y_pred[:,0:1]
    difference_mat_var = K.variable(difference_mat)
    difference_mat_var = tf.slice(difference_mat_var, [0, 0], K.shape(square_mat))
    difference_mat_var = difference_mat_var[:-1,:]
    return K.mean(K.square(y_pred - y_true), axis=-1) + reg_weight * K.mean(K.square(tf.matmul(difference_mat_var, y_flat)))

def main():
    from sys import argv
    argv = argv[1:]
    key = argv[0]
    #print(argv)
    assert key[0] == "-"
    if key == "-r":
        print("running regularization experiment")
        # regularization experiment
        mus_x_train, rec_x_train, core_train_features, y_train = util.load_data()
        run_reg_experiment(mus_x_train, rec_x_train, core_train_features, y_train)
    elif key == "-l":
        print("running linear model experiment")
        # linear model experiment
        n_features = int(argv[1])
        mus_x_train, rec_x_train, core_train_features, y_train = util.load_data(core_input_shape=n_features)
        run_linear(core_train_features, y_train, input_shape=n_features)
    elif key == "-n":
        print("running neural network experiment")
        # neural network
        mus_x_train, rec_x_train, core_train_features, y_train = util.load_data()
        run(mus_x_train, rec_x_train, core_train_features, y_train)
    else:
        print("Error, unrecognized case")


def run(mus_x_train, rec_x_train, core_train_features, y_train):
    init = TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    mus_input = Input(shape=(2 * util.TIMESTEPS + 1, 4))
    rec_input = Input(shape=(util.TIMESTEPS + 1, 2))

    reg_weight = 0.001

    mus = Dense(64, activation='relu',
                kernel_regularizer=reg.l2(reg_weight), bias_regularizer=reg.l2(reg_weight))(mus_input)
    rec = Dense(64, activation='relu',
                kernel_regularizer=reg.l2(reg_weight), bias_regularizer=reg.l2(reg_weight))(rec_input)

    concat = layers.concatenate([layers.Flatten()(mus), layers.Flatten()(rec)], axis=-1)
    mus_rec_dense = (Dense(128, activation='relu', kernel_initializer=init, name='combined_layer_1')(concat))

    mus_rec_dense3 = (Dense(32, activation='relu', kernel_initializer=init, name='combined_layer_3',
                   kernel_regularizer=reg.l2(reg_weight), bias_regularizer=reg.l2(reg_weight))(mus_rec_dense))

    output = Dense(2, activation='linear', kernel_initializer=init, name='output',
                   kernel_regularizer=reg.l2(reg_weight), bias_regularizer=reg.l2(reg_weight))(mus_rec_dense3)

    model = Model(inputs=[mus_input, rec_input], outputs=[output])

    model.compile(loss=custom_loss, optimizer='adam', metrics=[])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    N_EPOCHS = 1000
    history = model.fit([mus_x_train, rec_x_train], [y_train], validation_split=0.1,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[early_stopping], shuffle=True)
    #print(model.get_layer("output").get_weights())

    model.save('enhanced_nn_model.h5')
    predict.predict(model, withOffset=True)

def run_linear(core_train_features, y_train, input_shape=5):
    init = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
    core_input = Input(shape=(input_shape,))

    output = Dense(2, activation='linear', kernel_initializer=init, name='output')(core_input)

    model = Model(inputs=[core_input], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    N_EPOCHS = 1000
    history = model.fit([core_train_features], [y_train], validation_split=0.1,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[early_stopping, reduce_lr], shuffle=True)
    predict.predict(model, core_input_shape=input_shape)


l2_param_var = K.variable(0., dtype='float32')
def run_reg_experiment(mus_x_train, rec_x_train, core_train_features, y_train, repeat=15):
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.0000000001)
    # early_stopping = EarlyStopping(monitor='loss', patience=10)
    model = init_model()
    initial_weights = model.get_weights()
    average_losses = []
    l2_param = 0.00001
    for t in range(10):
        loss = []
        val_loss = []
        for i in range(repeat):
            np.random.seed(datetime.datetime.now().microsecond)
            shuffle_weights(model, weights=initial_weights)
            N_EPOCHS = 300
            K.set_value(l2_param_var, K.cast_to_floatx(l2_param))
            history = model.fit([mus_x_train, rec_x_train, core_train_features], [y_train], validation_split=0.1,
                                batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[],
                                shuffle=True,
                                verbose=True)
            if history.history['loss'][-1] > 50 or history.history['val_loss'][-1] > 50:
                continue
            loss.append(history.history['loss'][-1])
            val_loss.append(history.history['val_loss'][-1])
            #print('{},{}'.format(history.history['loss'][-1], history.history['val_loss'][-1]))
        average_losses.append((np.mean(loss), np.mean(val_loss)))
        l2_param *= 10
    print(average_losses)


def custom_l2(weights):
    return K.sum(l2_param_var * K.square(weights))

def init_model(l2_param=0.):
    init = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
    mus_input = Input(shape=(2 * util.TIMESTEPS + 1, 4))
    rec_input = Input(shape=(util.TIMESTEPS + 1, 2))
    core_features = Input(shape=(5,))

    concat = layers.concatenate([layers.Flatten()(mus_input), layers.Flatten()(rec_input)], axis=-1)

    mus_rec_dense = (Dense(32, activation='relu', kernel_initializer=init, name='combined_layer_1')(concat))
    mus_rec_dense2 = (Dense(1, activation='relu', kernel_initializer=init, name='combined_layer_2')(mus_rec_dense))
    output1 = Dense(8, activation='relu', kernel_initializer=init, name='output1')(layers.concatenate([mus_rec_dense2, core_features], axis=-1))

    output = Dense(2, activation='linear', kernel_initializer=init, name='output', kernel_regularizer=custom_l2)(output1)
    model = Model(inputs=[mus_input, rec_input, core_features], outputs=[output])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])

    #model.save('enhanced_nn_model.h5')


    #predict.predict(model)

    return model

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ == '__main__':
    main()






    '''
    init = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
    core_features = Input(shape=(5,))

    output = Dense(2, activation='linear', kernel_initializer=init, name='output')(core_features)
    model = Model(inputs=[core_features], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    tensorboard = TensorBoard(log_dir='./Graph/enhanced', histogram_freq=10, write_grads=True, write_graph=True,
                              write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    N_EPOCHS = 250
    model.fit([core_train_features], [y_train], validation_split=0.1,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[tensorboard, reduce_lr], shuffle=True)

    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    mus_input = Input(shape=(2 * util.TIMESTEPS + 1, 4))
    rec_input = Input(shape=(util.TIMESTEPS + 1, 2))
    core_input = Input(shape=(5,))

    # mus layers
    mus_conv = layers.Conv1D(16, 4, activation='relu',
                             input_shape=(2 * util.TIMESTEPS + 1, 4), kernel_initializer=init)(mus_input)
    mus_pool = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')(mus_conv)
    mus_dense = (Dense(16, activation='relu', kernel_initializer=init)(mus_pool))
    #rec layers
    rec_conv = layers.Conv1D(16, 4, activation='relu',
                             input_shape=(2 * util.TIMESTEPS + 1, 4), kernel_initializer=init)(rec_input)
    rec_pool = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')(rec_conv)
    rec_dense = (Dense(16, activation='relu', kernel_initializer=init)(rec_pool))

    concat = layers.concatenate([layers.Flatten()(mus_dense), layers.Flatten()(rec_dense)], axis=-1)

    mus_rec_dense = (Dense(128, activation='relu', kernel_initializer=init, name='combined_layer_1')(concat))
    mus_rec_dense2 = (Dense(64, activation='relu', kernel_initializer=init, name='combined_layer_2')(mus_rec_dense))
    theta = layers.concatenate([mus_rec_dense2, core_input], axis=-1)
    output = Dense(2, activation='linear', kernel_initializer=init, name='output')(theta)
    model = Model(inputs=[mus_input, rec_input, core_input], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    tensorboard = TensorBoard(log_dir='./Graph/enhanced', histogram_freq=10, write_grads=True, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    N_EPOCHS = 500
    model.fit([mus_x_train, rec_x_train, core_train_features], [y_train], validation_split=0.1,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[tensorboard, early_stopping, reduce_lr], shuffle=True)
    return model'''