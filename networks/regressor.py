from tf_siren import SinusodialRepresentationDense
from tf_siren import SIRENModel
import tensorflow as tf
from tf.keras.regularizes import l1, l2
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, LSTM
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


def regressor_network_siren(data_shape):
    def sine(x):
        return tf.math.sin(x)
    get_custom_objects().update({'sine':Activation(sine)})
    regressor1 = Sequential()
    regressor1.add(SinusodialRepresentationDense(53,input_dim=data_shape[1], kernel_initializer='he_uniform',activation='sine',kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)))
    regressor1.add(Dropout(0.1))
    regressor1.add(SinusodialRepresentationDense(25, kernel_initializer='he_uniform',activation='sine'))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))


def regressor_network(data_shape):
    def sine(x):
            return tf.math.sin(x)
    get_custom_objects().update({'sine':Activation(sine)})
    regressor1 = Sequential()
    regressor1.add(Dense(53,input_dim=data_shape[1], kernel_initializer='he_uniform',activation='sine',kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(25, kernel_initializer='he_uniform',activation='sine'))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))