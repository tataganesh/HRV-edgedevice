from tf_siren import SinusodialRepresentationDense
from tf_siren import SIRENModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
regressor1 = Sequential()
regressor1.add(Dense(114,input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation=tf.math.sin,kernel_regularizer=l2(0.01),activity_regularizer=l2(0.01)))
regressor1.add(Dropout(0.1))
# regressor1.add(Dense(2500, kernel_initializer='he_uniform', activation=tf.math.sin))
# regressor1.add(Dropout(0.25))
# regressor1.add(Dense(625, kernel_initializer='he_uniform', activation=tf.math.sin))
# regressor1.add(Dropout(0.25))
regressor1.add(Dense(50, kernel_initializer='he_uniform', activation=tf.math.sin))
regressor1.add(Dropout(0.1))
regressor1.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))