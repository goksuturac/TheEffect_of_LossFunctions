import tensorflow as tf
from tensorflow.keras import layers, models

def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    # Encoder_1
    x1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    # Encoder_2
    x2 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    # Encoder_3
    x3 = layers.MaxPooling2D((2, 2))(x2)
    x3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    # Encoder_4
    x4 = layers.MaxPooling2D((2, 2))(x3)
    x4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    
    # Bridge
    x5 = layers.MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.ReLU()(x5)
    x5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.ReLU()(x5)


    # Decoder_1
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x5)
    x = layers.concatenate([x, x4])
    x = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder_2
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x3])
    x = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Decoder_3
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x2])
    x = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Decoder_4
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.concatenate([x, x1], axis=3)
    x = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model