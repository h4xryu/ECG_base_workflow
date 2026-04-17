import tensorflow as tf


class ChannelAttention(tf.keras.layers.Layer):
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


class CATNet(tf.keras.layers.Layer):
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
