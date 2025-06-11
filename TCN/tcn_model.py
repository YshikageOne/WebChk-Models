import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate = 0.2):
    #residual block with dilated convulution
    input_x = x

    #dilated convolution
    x = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size,
                               dilation_rate=dilation_rate, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    #2nd convulution
    x = tf.keras.layers.Conv1D(filters = nb_filters, kernel_size = kernel_size, padding = 'same')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    #adjust residual connection if needed
    if input_x.shape[-1] != nb_filters:
        input_x = tf.keras.layers.Conv1D(nb_filters, kernel_size = 1, padding = 'same')(input_x)

    res = tf.keras.layers.Add()([input_x, x])
    res = tf.keras.layers.Activation('relu')(res)

    return res

def build_tcnModel(input_shape, num_classes, nb_filters = 64, kernel_size = 3, dropout_rate = 0.2):
    inputs = tf.keras.layers.Input(shape = input_shape)
    x = inputs

    #intial convulution
    conv1 = tf.keras.layers.Conv1D(nb_filters, kernel_size = 1, padding = 'same')(x)
    x = conv1

    #stack residual blocks
    dilation_rates = [1, 2, 4, 8]
    for dilation_rate in dilation_rates:
        x = residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate)

    #global pooling
    x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model





