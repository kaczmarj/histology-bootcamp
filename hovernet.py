"""HoVerNet implemented in tensorflow/keras."""

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers


class PreActResidualUnit(tfkl.Layer):
    """Pre-activated residual unit."""

    def __init__(self, filters, strides, activation="relu", **kwds):
        super().__init__(**kwds)

        self.filters = filters
        self.strides = strides
        self.activation = activation

        conv_kwds = dict(use_bias=False, padding="same")

        self.bn0 = tfkl.BatchNormalization()
        self.act0 = tfkl.Activation(self.activation)
        self.conv0 = tfkl.Conv2D(filters // 4, (1, 1), strides=strides, **conv_kwds)

        self.bn1 = tfkl.BatchNormalization()
        self.act1 = tfkl.Activation(self.activation)
        self.conv1 = tfkl.Conv2D(filters // 4, (3, 3), strides=1, **conv_kwds)

        self.bn2 = tfkl.BatchNormalization()
        self.act2 = tfkl.Activation(self.activation)
        self.conv2 = tfkl.Conv2D(filters, (1, 1), strides=1, **conv_kwds)

        # The use of this convolution depends on the shape of the inputs.
        self.skipconv = tfkl.Conv2D(filters, (1, 1), strides=strides, **conv_kwds)

    def call(self, inputs):
        # The skip logic (i.e., whether or not we apply an extra convolution)
        # must go in this caller because we have access to the input shape here.
        # We do not have input shape in init.
        if inputs.shape[-1] != self.filters or self.strides != 1:
            skip = self.skipconv(inputs)
        else:
            skip = inputs

        x = self.bn0(inputs)
        x = self.act0(x)
        x = self.conv0(x)

        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)

        x = tfkl.Add()([skip, x])
        return x


class PreActResidualBlock(tfkl.Layer):
    """Pre-activated residual block, composed of multiple
    pre-activated residual units.
    """

    def __init__(
        self, num_units: int, filters: int, strides: int, activation="relu", **kwds
    ):
        super().__init__(**kwds)
        self.num_units = num_units

        name = kwds.get("name")
        if name is not None:
            prefix = f"{name}/"
        else:
            prefix = ""
        # First layer can have strides!=1, but all other layers have strides=1.
        all_strides = [strides] + [1] * (num_units - 1)
        self.units = [
            PreActResidualUnit(
                filters=filters,
                strides=s,
                activation=activation,
                name=f"{prefix}unit_{i}",
            )
            for i, s in enumerate(all_strides)
        ]

    def call(self, inputs):
        x = inputs
        for unit_layer in self.units:
            x = unit_layer(x)
        return x


class DenseDecoderUnit(tfkl.Layer):
    def __init__(self, activation="relu", groups=1, **kwds):
        super().__init__(**kwds)
        self.activation = activation
        self.groups = groups

        self.bn0 = tfkl.BatchNormalization()
        self.act0 = tfkl.Activation(activation)
        self.conv0 = tfkl.Conv2D(
            128, (1, 1), strides=1, padding="valid", use_bias=False
        )
        self.bn1 = tfkl.BatchNormalization()
        self.act1 = tfkl.Activation(activation)
        self.conv1 = tfkl.Conv2D(
            # Input channels and filters must both be divisible by groups
            32,
            (5, 5),
            strides=1,
            padding="valid",
            use_bias=False,
            groups=self.groups,
        )

    def call(self, inputs):
        x = self.bn0(inputs)
        x = self.act0(x)
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        # TODO: this only supports channels_last format.
        # This crops a central portion of the inputs to
        # match the desired output shape.
        if inputs.shape[1:3] != x.shape[1:3]:
            h_orig, w_orig = inputs.shape[1:3]
            h_new, w_new = x.shape[1:3]
            h_crop = (h_orig - h_new) // 2
            w_crop = (w_orig - w_new) // 2
            # Symmetric crop of height and width.
            # For example, with input of shape (?, 270, 270, ?)
            # and desired output shape (?, 266, 266, ?), the
            # correct cropping is (2, 2).
            cropping = (h_crop, w_crop)
            shortcut = tfkl.Cropping2D(cropping)(inputs)
        else:
            shortcut = inputs

        x = tfkl.Concatenate(axis=-1)([shortcut, x])

        return x

    def get_config(self):
        return {"activation": self.activation, "groups": self.groups}


def get_center_cropping(input_shape, output_shape):
    """Get symmetric cropping values for CroppingND layer."""
    if len(input_shape) != len(output_shape):
        raise ValueError("input and output shapes must have same length")
    res = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]
    if any(n < 0 for n in res):
        raise ValueError("output shape cannot be larger than input shape")
    return res


# TODO: should this be a model or layer subclass?
class Decoder(tfkl.Layer):
    def __init__(self, num_classes: int, **kwds):
        super().__init__(**kwds)

        self.num_classes = num_classes

        self.upsample0 = tfkl.UpSampling2D(interpolation="nearest")
        self.conv00 = tfkl.Conv2D(256, (5, 5), use_bias=False, padding="valid")
        self.denseunits0 = [DenseDecoderUnit(groups=4) for _ in range(8)]
        self.conv01 = tfkl.Conv2D(512, (1, 1), padding="same")

        self.upsample1 = tfkl.UpSampling2D(interpolation="nearest")
        self.conv10 = tfkl.Conv2D(128, (5, 5), use_bias=False, padding="valid")
        self.denseunits1 = [DenseDecoderUnit(groups=4) for _ in range(4)]
        self.conv11 = tfkl.Conv2D(256, (1, 1), padding="same")

        self.upsample2 = tfkl.UpSampling2D(interpolation="nearest")
        # or valid padding?
        self.conv20 = tfkl.Conv2D(64, (5, 5), use_bias=False, padding="same")

        self.bn30 = tfkl.BatchNormalization()
        self.act30 = tfkl.Activation("relu")
        self.conv30 = tfkl.Conv2D(num_classes, (1, 1), use_bias=True, padding="valid")

    def call(self, inputs, *, encoder0, encoder1, encoder2):
        # Skip connections resemble U-Net's. Outputs of earlier layers
        # are connected to later layers.
        x = self.upsample0(inputs)
        crops = get_center_cropping(encoder2.shape[1:-1], output_shape=x.shape[1:-1])
        if crops != [0, 0]:
            encoder2 = tfkl.Cropping2D(crops)(encoder2)
        x = tfkl.Add()([x, encoder2])
        x = self.conv00(x)
        for denseunit in self.denseunits0:
            x = denseunit(x)
        x = self.conv01(x)

        x = self.upsample1(x)
        crops = get_center_cropping(
            input_shape=encoder1.shape[1:-1], output_shape=x.shape[1:-1]
        )
        encoder1 = tfkl.Cropping2D(crops)(encoder1)
        x = tfkl.Add()([x, encoder1])
        x = self.conv10(x)
        for denseunit in self.denseunits1:
            x = denseunit(x)
        x = self.conv11(x)

        x = self.upsample0(x)
        crops = get_center_cropping(encoder0.shape[1:-1], output_shape=x.shape[1:-1])
        encoder0 = tfkl.Cropping2D(crops)(encoder0)
        x = tfkl.Add()([x, encoder0])
        x = self.conv20(x)

        x = self.bn30(x)
        x = self.act30(x)
        x = self.conv30(x)

        return x


def hovernet(num_classes: int = 3, input_shape=(270, 270, 3)):
    inputs = tfkl.Input(shape=input_shape, name="inputs")

    # Pre-activated resnet encoder.
    x = tfkl.Conv2D(
        64,
        (7, 7),
        strides=1,
        padding="valid",
        use_bias=False,
        name="conv0",
    )(inputs)

    encoder0 = PreActResidualBlock(
        num_units=3, filters=256, strides=1, name="resblock_0"
    )(x)
    encoder1 = PreActResidualBlock(
        num_units=4, filters=512, strides=2, name="resblock_1"
    )(encoder0)
    encoder2 = PreActResidualBlock(
        num_units=6, filters=1024, strides=2, name="resblock_2"
    )(encoder1)

    encoder3 = PreActResidualBlock(
        num_units=3, filters=2048, strides=2, name="resblock_3"
    )(encoder2)

    x = tfkl.Conv2D(
        1024,
        (1, 1),
        use_bias=False,
        # TensorFlow 1.x implementation uses "same" padding but
        # Pytorch implementation uses 0 padding.
        padding="same",
        name="conv_bottleneck",
    )(encoder3)

    # This is the end of the preact resnet encoder.
    # Now we move on to the three decoder heads.

    nuclear_pixel = Decoder(num_classes=2, name="nuclear_pixel_head")(
        x, encoder0=encoder0, encoder1=encoder1, encoder2=encoder2
    )
    hover = Decoder(num_classes=2, name="hover_head")(
        x, encoder0=encoder0, encoder1=encoder1, encoder2=encoder2
    )
    tp = Decoder(num_classes=num_classes, name="tp_head")(
        x, encoder0=encoder0, encoder1=encoder1, encoder2=encoder2
    )
    return tfk.Model(inputs, outputs=[nuclear_pixel, hover, tp])
