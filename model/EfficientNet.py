import tensorflow as tf
from tensorflow.keras import layers, models

def build_EfficientNet(config):
    input_shape = tuple(config["input_shape"])
    inputs = layers.Input(shape=input_shape) # (512, 512, 1)

    x_rgb = layers.Concatenate()([inputs, inputs, inputs])  # (512, 512, 3)

    # Load EfficientNetB3 encoder
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x_rgb
    )

    # Skip connections
    skip1 = base_model.get_layer("block2a_activation").output  # (128, 128, 48)
    skip2 = base_model.get_layer("block3a_activation").output  # (64, 64, 64)
    skip3 = base_model.get_layer("block4a_activation").output  # (32, 32, 136)
    skip4 = base_model.get_layer("block6a_activation").output  # (16, 16, 232)

    encoder_output = base_model.get_layer("top_activation").output  # (16, 16, 384)

    # Decoder path
    decoder_config = config["DecoderCNN"]
    x = layers.Concatenate()([encoder_output, skip4])              # (16, 16, 616)
    x = layers.Conv2D(256, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (16, 16, 256)
    x = layers.Conv2D(256, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (16, 16, 256)

    x = layers.Conv2DTranspose(136, (2, 2), strides=(2, 2), padding="same")(x)  # (32, 32, 136)
    x = layers.Concatenate()([x, skip3])  # (32, 32, 272)
    x = layers.Conv2D(136, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (32, 32, 136)
    x = layers.Conv2D(136, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (32, 32, 136)

    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)  # (64, 64, 64)
    x = layers.Concatenate()([x, skip2])  # (64, 64, 128)
    x = layers.Conv2D(64, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (64, 64, 64)
    x = layers.Conv2D(64, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (64, 64, 64)

    x = layers.Conv2DTranspose(48, (2, 2), strides=(2, 2), padding="same")(x)  # (128, 128, 48)
    x = layers.Concatenate()([x, skip1])  # (128, 128, 96)
    x = layers.Conv2D(48, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (128, 128, 48)
    x = layers.Conv2D(48, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (128, 128, 48)

    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(x)  # (256, 256, 32)
    x = layers.Conv2D(32, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (256, 256, 32)
    x = layers.Conv2D(32, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (256, 256, 32)

    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(x)  # (512, 512, 16)
    x = layers.Conv2D(16, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (512, 512, 16)
    x = layers.Conv2D(16, kernel_size=tuple(decoder_config["kernel_size"]), padding='same', activation=decoder_config["activation"])(x)  # (512, 512, 16)

    output_config = config["OutputLayer"]
    outputs = layers.Conv2D(output_config["filters"], (1, 1), activation=output_config["activation"])(x)  # (512, 512, 1)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model