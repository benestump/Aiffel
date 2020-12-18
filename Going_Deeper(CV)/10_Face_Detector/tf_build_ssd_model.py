import tensorflow as tf

def SsdModel(cfg, num_cell, training=False, name='ssd_model'):
    image_sizes = cfg['input_size']   if training else None
    if isinstance(image_sizes, int):
        image_sizes = (image_sizes, image_sizes)
    elif isinstance(image_sizes, tuple):
        image_sizes = image_sizes
    elif image_sizes == None:
        image_sizes = (None, None)
    else:
        raise Exception('Type error of input image size format,tuple or int. ')

    base_channel = cfg["base_channel"]
    num_class = len(cfg['labels_list'])

    x = inputs = tf.keras.layers.Input(shape=[image_sizes[0], image_sizes[1], 3], name='input_image')

    x = _conv_block(x, base_channel, strides=(2, 2))  # 120*160*16
    x = _conv_block(x, base_channel * 2, strides=(1, 1))
    x = _conv_block(x, base_channel * 2, strides=(2, 2))  # 60*80
    x = _conv_block(x, base_channel * 2, strides=(1, 1))
    x = _conv_block(x, base_channel * 4, strides=(2, 2))  # 30*40
    x = _conv_block(x, base_channel * 4, strides=(1, 1))
    x = _conv_block(x, base_channel * 4, strides=(1, 1))
    x = _conv_block(x, base_channel * 4, strides=(1, 1))
    x1 = _branch_block(x, base_channel)

    x = _conv_block(x, base_channel * 8, strides=(2, 2))  # 15*20
    x = _conv_block(x, base_channel * 8, strides=(1, 1))
    x = _conv_block(x, base_channel * 8, strides=(1, 1))
    x2 = _branch_block(x, base_channel)

    x = _depthwise_conv_block(x, base_channel * 16, strides=(2, 2))  # 8*10
    x = _depthwise_conv_block(x, base_channel * 16, strides=(1, 1))
    x3 = _branch_block(x, base_channel)

    x = _depthwise_conv_block(x, base_channel * 16, strides=(2, 2))  # 4*5
    x4 = _branch_block(x, base_channel)

    extra_layers = [x1, x2, x3, x4]

    confs = []
    locs = []

    head_idx = 0
    assert len(extra_layers) == len(num_cell)
    for layer in extra_layers:
        conf, loc = _compute_heads(layer, head_idx, num_class, num_cell)
        confs.append(conf)
        locs.append(loc)

        head_idx += 1

    confs = tf.keras.layers.Concatenate(axis=1, name="face_classes")(confs)
    locs = tf.keras.layers.Concatenate(axis=1, name="face_boxes")(locs)

    predictions = tf.keras.layers.Concatenate(axis=2, name='predictions')([locs, confs])


    model = tf.keras.Model(inputs=inputs, outputs=predictions, name=name)
    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), use_bn=True, padding=None, block_id=None):
    """Adds an initial convolution layer (with batch normalization and relu).
    # Returns
        Output tensor of block.
    """
    if block_id is None:
        block_id = (tf.keras.backend.get_uid())

    if strides == (2, 2):
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv_pad_%d' % block_id)(inputs)
        x = tf.keras.layers.Conv2D(filters, kernel,
                                   padding='valid',
                                   use_bias=False if use_bn else True,
                                   strides=strides,
                                   name='conv_%d' % block_id)(x)
    else:
        x = tf.keras.layers.Conv2D(filters, kernel,
                                   padding='same',
                                   use_bias=False if use_bn else True,
                                   strides=strides,
                                   name='conv_%d' % block_id)(inputs)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='conv_bn_%d' % block_id)(x)
    return tf.keras.layers.ReLU(name='conv_relu_%d' % block_id)(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), use_bn=True, block_id=None):
    """Adds a depthwise convolution block.
        # Returns
        Output tensor of block.
    """
    if block_id is None:
        block_id = (tf.keras.backend.get_uid())

    if strides == (1, 1):
        x = inputs
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), name='conv_pad_%d' % block_id)(inputs)

    x = tf.keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False if use_bn else True,
                                        name='conv_dw_%d' % block_id)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = tf.keras.layers.ReLU(name='conv_dw_%d_relu' % block_id)(x)

    x = tf.keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                               padding='same',
                               use_bias=False if use_bn else True,
                               strides=(1, 1),
                               name='conv_pw_%d' % block_id)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return tf.keras.layers.ReLU(name='conv_pw_%d_relu' % block_id)(x)


def _branch_block(input, filters):

    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(x)

    x1 = tf.keras.layers.Conv2D(filters * 2, kernel_size=(3, 3), padding='same')(input)

    x = tf.keras.layers.Concatenate(axis=-1)([x, x1])

    return tf.keras.layers.ReLU()(x)


def _create_head_block(inputs, filters, strides=(1, 1), block_id=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
    return x

def _compute_heads(x, idx, num_class, num_cell):
    """ Compute outputs of classification and regression heads
    Args:
        x: the input feature map
        idx: index of the head layer
    Returns:
        conf: output of the idx-th classification head
        loc: output of the idx-th regression head
    """
    conf = _create_head_block(inputs=x, filters=num_cell[idx] * num_class)
    conf = tf.keras.layers.Reshape((-1, num_class))(conf)
    loc = _create_head_block(inputs=x, filters=num_cell[idx] * 4)
    loc = tf.keras.layers.Reshape((-1, 4))(loc)

    return conf, loc


if __name__ == '__main__':
    import config
    import os

    cfg = config.cfg
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = SsdModel(cfg, num_cell=[3, 2, 2, 3],training=False)
    print(len(model.layers))
    model.summary()
