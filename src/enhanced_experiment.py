from src import util
from keras.models import Model
from keras.layers import Dense, Input
import keras.layers as layers
from keras.initializers import TruncatedNormal
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
import keras.backend as K

BATCH_SIZE = 512

def custom_loss(y_true, y_pred):
    condition = K.cast(K.greater_equal(y_pred, y_true), 'float32')
    loss = 0
    loss += K.sum(condition)
    print(condition.get_shape())
    loss /= K.count_params(condition)
    loss *= K.mean(K.dot(condition, K.abs(y_pred - y_true)), axis=-1)
    not_condition = K.cast(K.less(y_pred, y_true), 'float32')
    loss += K.sum(not_condition) / K.count_params(not_condition) * K.mean(K.dot(not_condition, K.exp(y_true - y_pred)), axis=-1)
    return loss

def main():
    musList, recList, matchesMapList, songNames = util.parseMatchedInput('javaOutput', 8)
    musList, recList = util.normalizeTimes(musList, recList)
    recList, matchesMapList = util.trim(recList, matchesMapList)
    x, y = util.dataAsWindow(musList, recList, matchesMapList)
    x_train = x.astype('float32')
    y_train = y.astype('float32')
    mus_train, rec_train, core_train = util.splitData(x_train)
    mus_train, rec_train, core_train = mus_train.astype('float32'), rec_train.astype('float32'), core_train.astype('float32')

    ################
    isSeq = False  ##
    ################

    mus_train, rec_train, core_train, y_train = util.splitSeqChord(mus_train, rec_train, core_train, y_train, isSeq)

    model = train_model(mus_train, rec_train, core_train, y_train)
    if isSeq:
        model.save("enhanced_nn_model_seq.h5")
    else:
        model.save("enhanced_nn_model_chord.h5")


def train_model(mus_x_train, rec_x_train, core_train_features, y_train):
    init = TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
    mus_input = Input(shape=(2 * util.TIMESTEPS + 1, 4))
    rec_input = Input(shape=(util.TIMESTEPS + 1, 2))
    core_features = Input(shape=(5,))

    concat = layers.concatenate([layers.Flatten()(mus_input), layers.Flatten()(rec_input)], axis=-1)

    mus_rec_dense = (Dense(32, activation='relu', kernel_initializer=init, name='combined_layer_1')(concat))
    mus_rec_dense2 = (Dense(1, activation='relu', kernel_initializer=init, name='combined_layer_2')(mus_rec_dense))
    output1 = Dense(8, activation='relu', kernel_initializer=init, name='output1')(
        layers.concatenate([mus_rec_dense2, core_features], axis=-1))
    output = Dense(2, activation='linear', kernel_initializer=init, name='output')(output1)
    model = Model(inputs=[mus_input, rec_input, core_features], outputs=[output])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    tensorboard = TensorBoard(log_dir='./Graph/enhanced', histogram_freq=10, write_grads=True, write_graph=True,
                              write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.0000000001)
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    N_EPOCHS = 300
    model.fit([mus_x_train, rec_x_train, core_train_features], [y_train], validation_split=0.1,
              batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[tensorboard, reduce_lr], shuffle=True)
    return model


# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()