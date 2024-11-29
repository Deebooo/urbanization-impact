import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate
)
from tensorflow.keras.optimizers import Adam

def unet(input_size=(512, 512, 3), metrics=None):
    inputs = Input(input_size)

    # Downsampling
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)

    # Bridge
    conv6 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    # Upsampling
    up7 = Conv2DTranspose(1024, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(drop6)
    merge7 = Concatenate()([drop5, up7])
    conv7 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = Concatenate()([drop4, up8])
    conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    up9 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = Concatenate()([conv3, up9])
    conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    up10 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv9)
    merge10 = Concatenate()([conv2, up10])
    conv10 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    
    up11 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv10)
    merge11 = Concatenate()([conv1, up11])
    conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    
    output = Conv2D(1, 1, activation='sigmoid')(conv11)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=metrics)

    return model