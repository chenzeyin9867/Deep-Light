import tensorflow as tf

def normalize(image):
    # image=ALPHA*(image**(1/GAMMA))-1
    # image = ALPHA * (image ** (1 / GAMMA))
    image = (image - 127.5) / 127.5
    # image=(image-trainDataMean)/trainDataStd
    return image


def deNormalize(image):
    # image=((image+1)/ALPHA)**GAMMA
    # image = (image / ALPHA) ** GAMMA
    image = image * 127.5 + 127.5
    # image=image*trainDataStd+trainDataMean
    return image



def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# In[17]:


# In[18]:


from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


# Conv2d+BatchNorm2d+ReLU
def cbr(input_tensor, filters, kernel_size, strides):
    #  x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
    #                    use_bias=False, kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = layers.Activation('relu')(x)
    return x

# Conv2d+BatchNorm2d
def cbr_NoRelu(input_tensor, filters, kernel_size, strides):
    #  x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
    #                    use_bias=False, kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)

    return x



def NewResidual_1(input_tensor,filters):
    x = input_tensor
    y = cbr(input_tensor,filters,3,2)
    y = cbr_NoRelu(y,filters,3,1)
    x = cbr_NoRelu(x,filters,1,2)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    x = y
    y = cbr(y,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    return y

def NewResidual_2(input_tensor,filters):
    x = NewResidual_1(input_tensor,filters)
    return x


def skyEncoder():
    inputs = tf.keras.layers.Input(shape=[32, 128, 3])
    initializer = tf.random_normal_initializer(0., 0.02)
    # result = tf.keras.Sequential()
    # result.add(
    #    tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
    #                           kernel_initializer=initializer, use_bias=False))
    result = tf.keras.layers.Conv2D(64, 7, strides=1, padding='same', activation=tf.nn.elu,
                                    kernel_initializer=initializer, use_bias=False)(inputs)
    # result = residual_1(result, 64)
    # result = residual_2(result, 16)
    result = NewResidual_1(result,64)
    result = NewResidual_2(result,16)
    result = tf.keras.layers.Flatten()(result)
    result = tf.keras.layers.Dense(64)(result);

    # result=result(inputs)
    return tf.keras.Model(inputs=inputs, outputs=result)

def NewResidual_3(input_tensor,filters):
    #1 block
    x = input_tensor
    y = cbr(input_tensor,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    x = cbr_NoRelu(x,filters,1,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    #2 block
    x = y
    y = cbr(y, filters, 3, 1)
    y = cbr_NoRelu(y, filters, 3, 1)
    y = layers.add([x, y])
    y = layers.Activation('relu')(y)
    # 3 block
    x = y
    y = cbr(y, filters, 3, 1)
    y = cbr_NoRelu(y, filters, 3, 1)
    y = layers.add([x, y])
    y = layers.Activation('relu')(y)
    return y


def NewResidual_4(input_tensor,filters):
    x = input_tensor
    y = cbr(input_tensor,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    #2nd block
    x = y
    y = cbr(y,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    return y


def skyDecoder():
    inputs = tf.keras.layers.Input(shape=[64])
    result = tf.keras.layers.Dense(4096)(inputs)
    result = layers.Activation('elu')(result)
    result = tf.keras.layers.Reshape([8, 32, 16])(result)
    result = upsample(16, 3)(result)
    # result = residual_3(result, 64)
    result = NewResidual_3(result,64)
    result = upsample(64, 3)(result)
    result = NewResidual_4(result, 64)
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(
        result)
    return tf.keras.Model(inputs=inputs, outputs=result)


# In[22]:
