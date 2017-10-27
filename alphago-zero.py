#!/usr/bin/python

import keras

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation

input_data = Input(shape=(19, 19, 17))

def conv_block(x):
    y = Conv2D(256, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def residual_block(x):
    y = Conv2D(256, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(256, (3, 3), padding='same')(y)
    y = BatchNormalization()(y)
    y = keras.layers.add([x, y])
    y = Activation('relu')(y)
    return y

def policy_head(x):
    y = Conv2D(2, (1, 1), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dense(19**2+1, activation='sigmoid')(y)
    return y

def value_head(x):
    y = Conv2D(1, (1, 1), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dense(256)(y)
    y = Activation('relu')(y)
    y = Dense(1)(y)
    y = Activation('tanh')(y)
    return y

# in the paper there were either 39 or 19 residual blocks
def alphago_zero_nn(residual_blocks=39):
    x = conv_block(input_data)

    for i in range(residual_blocks):
        x = residual_block(x)
    
    policy_out = policy_head(x)
    value_out = value_head(x)

    model = Model(inputs=[input_data], outputs=[policy_out, value_out])

    return model


model_alphago_zero = alphago_zero_nn()
print(model_alphago_zero.summary())



