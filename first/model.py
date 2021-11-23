# import numpy as np
# A = np.load('data_train.npz')
# print(A['X_en_tra'])

from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
import tensorflow.keras
import numpy as np

MAX_LEN_en = 200
NB_WORDS = 16385
# NB_WORDS = 65
# NB_WORDS = 257
# NB_WORDS = 1025
# NB_WORDS = 4097
# NB_WORDS = 16385
# NB_WORDS = 65537
EMBEDDING_DIM = 100
embedding_matrix = np.load('D:/pycharm_pro/My-Enhancer-classification/embedding/embedding_matrix7.npy')
embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weight = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_model():
    enhancers = Input(shape=(MAX_LEN_en,))
    emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True)(enhancers)
    enhancer_conv_layer = Convolution1D(
                                        filters=256,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer2 = Convolution1D(
                                        filters=128,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer2 = MaxPooling1D(pool_size=int(2))
    # enhancer_conv_layer3 = Convolution1D(
    #                                     filters=64,
    #                                     kernel_size=8,
    #                                     padding="same",  # "same"
    #                                     )
    # enhancer_max_pool_layer3 = MaxPooling1D(pool_size=int(2))
    # Build enhancer branch
    enhancer_branch = Sequential()
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_branch.add(enhancer_conv_layer2)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer2)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    # enhancer_branch.add(enhancer_conv_layer3)
    # enhancer_branch.add(Activation("relu"))
    # enhancer_branch.add(enhancer_max_pool_layer3)
    # enhancer_branch.add(BatchNormalization())
    # enhancer_branch.add(Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)


    # l_gru_1 = Bidirectional(GRU(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru_1)
    # bn = BatchNormalization()(l_att_1)
    # dt = Dropout(0.2)(bn)

    l_gru = Bidirectional(LSTM(64, return_sequences=True))(enhancer_out)
    l_att = AttLayer(64)(l_gru)
    # l_gru = Bidirectional(SimpleRNN(64, return_sequences=True))(enhancer_out)
    # l_att = AttLayer(64)(l_gru)
    bn2 = BatchNormalization()(l_att)
    dt2 = Dropout(0.2)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(64,kernel_initializer="glorot_uniform")(dt2)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([enhancers], preds)
    adam = tensorflow.keras.optimizers.Adam(lr=4e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
    # 5e-6(测试集)  4e-5(训练集)

def get_model_onehot():
    enhancers = Input(shape=(MAX_LEN_en,))
    emb_en = Embedding(5, 4, weights=[embedding_matrix_one_hot],
                                  trainable=False)(enhancers)
    enhancer_conv_layer = Convolution1D(
                                        filters=256,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer = MaxPooling1D(pool_size=int(2))
    enhancer_conv_layer2 = Convolution1D(
                                        filters=128,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        )
    enhancer_max_pool_layer2 = MaxPooling1D(pool_size=int(2))

    # Build enhancer branch
    enhancer_branch = Sequential()
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_branch.add(enhancer_conv_layer2)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer2)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.2))
    enhancer_out = enhancer_branch(emb_en)


    # l_gru_1 = Bidirectional(GRU(64, return_sequences=True))(enhancer_out)
    # l_att_1 = AttLayer(64)(l_gru_1)
    # bn = BatchNormalization()(l_att_1)
    # dt = Dropout(0.2)(bn)

    l_gru = Bidirectional(LSTM(64, return_sequences=True))(enhancer_out)
    l_att = AttLayer(64)(l_gru)
    bn2 = BatchNormalization()(l_att)
    dt2 = Dropout(0.2)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(64,kernel_initializer="glorot_uniform")(dt2)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.2)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([enhancers], preds)
    adam = tensorflow.keras.optimizers.Adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model


def get_model_1():
    l_att = Input(shape=(64,))
    preds = Dense(1, activation='sigmoid')(l_att)
    model = Model([l_att], preds)
    adam = tensorflow.keras.optimizers.Adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model

def get_model_2():
    enhancers = Input(shape=(MAX_LEN_en,))
    model = Sequential()
    model.add(enhancers)
    model.add(Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       embedding_matrix], trainable=True))
    model.add(Convolution1D(
                                        filters=256,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        ))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=int(2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution1D(
                                        filters=128,
                                        kernel_size=8,
                                        padding="same",  # "same"
                                        ))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=int(2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(AttLayer(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64,kernel_initializer="glorot_uniform"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    adam = tensorflow.keras.optimizers.Adam(lr=4e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    return model
    # 5e-6(测试集)  4e-5(训练集)


