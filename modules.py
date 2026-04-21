import tensorflow as tf

keras = tf.keras


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='Custom')
class ConvBNLeaky(keras.layers.Layer):
    """Conv1D → BatchNormalization → LeakyReLU  (B, T, C)."""

    def __init__(self, filters, kernel_size=9, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU()

    def call(self, x, training=False):
        return self.act(self.bn(self.conv(x), training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'kernel_size': self.kernel_size,
                    'strides': self.strides})
        return cfg


@keras.saving.register_keras_serializable(package='Custom')
class TransConvBNLeaky(keras.layers.Layer):
    """Conv1DTranspose → BatchNormalization → LeakyReLU  (B, T, C)."""

    def __init__(self, filters, kernel_size=9, **kwargs):
        super().__init__(**kwargs)
        self.filters     = filters
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1DTranspose(
            filters, kernel_size, strides=2, padding='same', use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.LeakyReLU()

    def call(self, x, training=False):
        return self.act(self.bn(self.conv(x), training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'kernel_size': self.kernel_size})
        return cfg


# ---------------------------------------------------------------------------
# U-Net encoder / decoder stacks
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='Custom')
class UNetEncoder(keras.layers.Layer):
    """Strided-conv encoder; returns (output, skip_list)."""

    def __init__(self, mid_ch, num_layers, kernel_size=9, **kwargs):
        super().__init__(**kwargs)
        self.mid_ch      = mid_ch
        self.num_layers  = num_layers
        self.kernel_size = kernel_size
        self.enc_layers  = [
            ConvBNLeaky(mid_ch, kernel_size, strides=2, name=f'enc_{i}')
            for i in range(num_layers)
        ]

    def call(self, x, training=False):
        skips = []
        for layer in self.enc_layers:
            x = layer(x, training=training)
            skips.append(x)
        return x, skips

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'mid_ch': self.mid_ch, 'num_layers': self.num_layers,
                    'kernel_size': self.kernel_size})
        return cfg


@keras.saving.register_keras_serializable(package='Custom')
class UNetDecoder(keras.layers.Layer):
    """Transposed-conv decoder consuming skip connections from the encoder."""

    def __init__(self, out_ch, mid_ch, num_layers, kernel_size=9, **kwargs):
        super().__init__(**kwargs)
        self.out_ch      = out_ch
        self.mid_ch      = mid_ch
        self.num_layers  = num_layers
        self.kernel_size = kernel_size
        self.dec_layers  = [
            TransConvBNLeaky(
                out_ch if i == num_layers - 1 else mid_ch,
                kernel_size, name=f'dec_{i}',
            )
            for i in range(num_layers)
        ]

    def call(self, x, skips, training=False):
        for i, layer in enumerate(self.dec_layers):
            skip = skips[-1 - i]
            x = x[:, :tf.shape(skip)[1], :]
            x = tf.concat([x, skip], axis=-1)
            x = layer(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'out_ch': self.out_ch, 'mid_ch': self.mid_ch,
                    'num_layers': self.num_layers, 'kernel_size': self.kernel_size})
        return cfg


# ---------------------------------------------------------------------------
# ResidualUBlock
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='Custom')
class ResidualUBlock(keras.layers.Layer):
    """
    Residual U-Net block for 1-D signals  (B, T, C).

    Args:
        out_ch:       Input/output channel count.
        mid_ch:       Hidden channels inside the U-Net arms.
        layers:       Number of encoder/decoder stages.
        downsampling: If True, AvgPool + 1×1 Conv applied at the end.
    """

    def __init__(self, out_ch, mid_ch, layers, downsampling=True, **kwargs):
        super().__init__(**kwargs)
        self.out_ch       = out_ch
        self.mid_ch       = mid_ch
        self.num_layers   = layers
        self.downsampling = downsampling
        K = 9

        self.entry_conv = tf.keras.layers.Conv1D(out_ch, K, padding='same', use_bias=False)
        self.entry_bn   = tf.keras.layers.BatchNormalization()
        self.entry_act  = tf.keras.layers.LeakyReLU()

        self.encoder    = UNetEncoder(mid_ch, layers, K)
        self.bottleneck = ConvBNLeaky(mid_ch, K)
        self.decoder    = UNetDecoder(out_ch, mid_ch, layers, K)

        if downsampling:
            self.pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)
            self.proj = tf.keras.layers.Conv1D(out_ch, kernel_size=1, use_bias=False)

    def call(self, x, training=False):
        x_in = self.entry_act(self.entry_bn(self.entry_conv(x), training=training))

        enc_out, skips = self.encoder(x_in, training=training)
        dec_out = self.decoder(
            self.bottleneck(enc_out, training=training), skips, training=training)

        out = dec_out[:, :tf.shape(x_in)[1], :] + x_in

        if self.downsampling:
            out = self.proj(self.pool(out))

        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'out_ch': self.out_ch, 'mid_ch': self.mid_ch,
                    'layers': self.num_layers, 'downsampling': self.downsampling})
        return cfg


# ---------------------------------------------------------------------------
# Original modules
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='Custom')
class ChannelAttention(keras.layers.Layer):
    """CBAM channel attention for 1-D feature maps (B, T, C)."""

    def __init__(self, filters, ratio, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio   = ratio
        mid = max(1, filters // ratio)
        self.fc1 = tf.keras.layers.Dense(mid, activation='relu',
                                         kernel_initializer='he_normal', use_bias=True)
        self.fc2 = tf.keras.layers.Dense(filters,
                                         kernel_initializer='he_normal', use_bias=True)

    def call(self, x):
        # Average-pool branch
        avg = tf.reduce_mean(x, axis=1)          # (B, C)
        avg = self.fc2(self.fc1(avg))             # (B, C)

        # Max-pool branch
        mx = tf.reduce_max(x, axis=1)            # (B, C)
        mx = self.fc2(self.fc1(mx))              # (B, C)

        scale = tf.sigmoid(avg + mx)             # (B, C)
        scale = tf.expand_dims(scale, axis=1)    # (B, 1, C)  — broadcast over T
        return x * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'ratio': self.ratio})
        return cfg


@keras.saving.register_keras_serializable(package='Custom')
class CATNet(keras.layers.Layer):
    """Conv-Attention-LSTM backbone (blocks 1-4 + recurrent)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Conv-Attention blocks
        self.conv1  = tf.keras.layers.Conv1D(16,  kernel_size=21, padding='SAME', activation='relu')
        self.attn1  = ChannelAttention(16,  ratio=8)
        self.pool1  = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME')

        self.conv2  = tf.keras.layers.Conv1D(32,  kernel_size=23, padding='SAME', activation='relu')
        self.attn2  = ChannelAttention(32,  ratio=8)
        self.pool2  = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME')

        self.conv3  = tf.keras.layers.Conv1D(64,  kernel_size=25, padding='SAME', activation='relu')
        self.attn3  = ChannelAttention(64,  ratio=8)
        self.pool3  = tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME')

        self.conv4  = tf.keras.layers.Conv1D(128, kernel_size=27, padding='SAME', activation='relu')
        self.attn4  = ChannelAttention(128, ratio=8)

        # Recurrent
        self.lstm1  = tf.keras.layers.LSTM(64, return_sequences=True)
        self.drop1  = tf.keras.layers.Dropout(0.2)
        self.lstm2  = tf.keras.layers.LSTM(32, return_sequences=True)

    def call(self, x, training=False):
        x = self.pool1(self.attn1(self.conv1(x)))
        x = self.pool2(self.attn2(self.conv2(x)))
        x = self.pool3(self.attn3(self.conv3(x)))
        x = self.attn4(self.conv4(x))
        x = self.lstm1(x, training=training)
        x = self.drop1(x, training=training)
        x = self.lstm2(x, training=training)
        return x

    def get_config(self):
        return super().get_config()
