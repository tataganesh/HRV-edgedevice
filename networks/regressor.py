from tf_siren import SinusodialRepresentationDense
from tf_siren import SIRENModel
from tensorflow import keras
from keras.regularizers import l1, l2
from  keras import layers
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, LSTM
from keras.layers import Activation
import tensorflow as tf

def regressor_network_siren(data_shape):
    regressor1 = Sequential()
    regressor1.add(SinusodialRepresentationDense(53,input_dim=data_shape[1], kernel_initializer='he_uniform',activation=tf.math.sin,kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)))
    regressor1.add(Dropout(0.1))
    regressor1.add(SinusodialRepresentationDense(25, kernel_initializer='he_uniform',activation=tf.math.sin))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))
    return regressor1

def regressor_network(data_shape):
    regressor1 = Sequential()
    regressor1.add(Dense(53,input_dim=data_shape[1], kernel_initializer='he_uniform',activation=tf.math.sin,kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(25, kernel_initializer='he_uniform',activation=tf.math.sin))
    regressor1.add(Dropout(0.1))
    regressor1.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))
    return regressor1