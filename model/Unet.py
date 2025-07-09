import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, kernel_size, activation, dropout, use_batchnorm):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def upsample_block(x, skip, filters, kernel_size, activation, dropout, use_batchnorm):
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, kernel_size, activation, dropout, use_batchnorm)
    return x


def build_Unet(config):
    input_shape = tuple(config["input_shape"])
    inputs = layers.Input(shape=input_shape)

    x = inputs
    skips = []

    # Encoder
    for block in config["EncoderCNN"]["conv_blocks"]:
        x = conv_block(
            x,
            filters=block["filters"],
            kernel_size=tuple(block["kernel_size"]),
            activation=block["activation"],
            dropout=block.get("dropout", 0.0),
            use_batchnorm=block.get("use_batchnorm", False)
        )
        skips.append(x)
        if block.get("maxpool", False):
            x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    bottleneck = config["Bottleneck"]
    x = conv_block(
        x,
        filters=bottleneck["filters"],
        kernel_size=tuple(bottleneck["kernel_size"]),
        activation=bottleneck["activation"],
        dropout=bottleneck.get("dropout", 0.0),
        use_batchnorm=bottleneck.get("use_batchnorm", False)
    )

    # Decoder
    for block, skip in zip(config["DecoderCNN"]["upsample_blocks"], reversed(skips)):
        x = upsample_block(
            x,
            skip,
            filters=block["filters"],
            kernel_size=tuple(block["kernel_size"]),
            activation=block["activation"],
            dropout=block.get("dropout", 0.0),
            use_batchnorm=block.get("use_batchnorm", False)
        )

    # Output layer
    output_config = config["OutputLayer"]
    outputs = layers.Conv2D(
        filters=output_config["filters"],
        kernel_size=(1, 1),
        activation=output_config["activation"]
    )(x)

    model = models.Model(inputs, outputs, name="U-Net")
    return model
