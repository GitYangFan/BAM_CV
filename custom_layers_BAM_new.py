import tensorflow as tf
import math
#Canges:
#BAM layer:
# - removed lambda
# - added values
#Activation layer:
# - different activation functions per channel
# - removed lambda
# - different initialization
# - BUGFIX: l1_constraintLessEqual set values to zero if they exceeded 1, replaced l1_constraintLessEqual by l1_constraintLessEqual2
# - different initialization: Identity

#Cleaned code


tf.keras.backend.set_floatx('float32')
ddtype = tf.float32


class model_attention_final(tf.keras.Model):
    """
    Main class, full model with observational attention and bivariate attention in the SPD space

    Args:
        n_channels_main (int): Number of channels in the main layer.
        data_layers (int): Number of layers for attention between features and samples.
        cov_layers (int): Number of layers for bilinear attention and activation on the covariance matrix.
        inner_channels (int): Number of inner channels in the model.
        N_exp (int): Number of exponentiations for the SPD activation.
        N_heads (int): Number of heads for the multi-head attention mechanisms.
    """

    def __init__(self, n_channels_main=10, data_layers=2, cov_layers=4, inner_channels=10, N_exp=3, N_heads=5,
                 num_classes=7):
        super(model_attention_final, self).__init__()
        self.data_layers = data_layers
        self.cov_layers = cov_layers
        self.inner_channels = inner_channels
        self.n_channels_main = n_channels_main
        self.N_exp = N_exp
        self.N_heads = N_heads
        self.num_classes = num_classes
        self.weight1 = tf.Variable(initial_value=1, trainable=True, name='weight1', dtype=tf.float32)
        self.weight2 = tf.Variable(initial_value=0, trainable=True, name='weight2', dtype=tf.float32)
        l: int
        for l in range(1, self.cov_layers + 1):
            setattr(self, f"layer_N_C_d_d_bilinear_attention{l}",
                    MultiHeadAttention_N_C_d_d_bilinear(num_heads=self.N_heads))
            setattr(self, f"layer_N_C_d_d_spd_activation{l}",
                    layer_N_C_d_d_spd_activation_scaled(N_exp=self.N_exp))
        self.layer_N_M_d_1_to_N_x_x_C_conv = layer_N_M_d_1_to_N_x_x_C_conv(out_filters=self.n_channels_main)
        self.layer_dense = layer_dense(self.num_classes)

    def call(self, inputs, **kwargs):
        out = tf.expand_dims(inputs, 3)

        # use the convolution layers to reduce the complexity
        # conv1 = self.layer_N_M_d_1_to_N_x_x_C_conv(o1)              # shape (N, k, k, C)
        # # compute the covariance matrices here
        # cov1 = data_N_M_d_c_to_cov_N_C2_C1_C1_image(conv1, 1)       # shape (N, 1, C, C)
        # cov1 = tf.transpose(cov1, [0, 2, 3, 1])  # shape (N, C, C, 1)

        # add some image attention layers here
        conv1 = self.layer_N_M_d_1_to_N_x_x_C_conv(out)  # reduce the complexity       # shape (N, k, k, C)
        # image_representation = tf.transpose(conv1, [0, 3, 1, 2])  # shape (N, C, k, k)

        cov_base = _cal_cov_pooling(conv1)  # shape (N, 1, c, c)
        out = cov_base  # model2: covariance matrix - shape (N, C2, C1, C1)
        for l in range(1, self.cov_layers + 1):
            out = getattr(self, f"layer_N_C_d_d_bilinear_attention{l}")(out)
            out = getattr(self, f"layer_N_C_d_d_spd_activation{l}")(out)
        out_reshape = tf.reshape(out, shape=(-1, 1, self.n_channels_main, self.n_channels_main))  # shape (N, 1, c, c)

        # here throw out softmax output and keep shape [N,width,width,C]
        # # option 1: baseline
        # cov_baseline = self.layer_baseline(conv1)
        # final_output = self.layer_dense(cov_baseline)

        # # option 2: BAM original softmax
        # fusion = feature_fusion(conv1, cov_euklidean, weight1=self.weight1, weight2=self.weight2)
        # final_output = self.layer_softmax2(fusion)

        # option 3: BAM modified LogEig with dense
        # cov_euklidean = cal_logeig(out)
        cov_euklidean = _cal_log_cov(out_reshape)
        shape_cov = tf.shape(cov_euklidean)
        cov_euklidean_reshape = tf.reshape(cov_euklidean, shape=(shape_cov[0], shape_cov[1], shape_cov[2], 1))  # shape (N, c, c, 1)
        fusion = feature_fusion(cov_euklidean_reshape, conv1, weight1=self.weight1, weight2=self.weight2)   # resize 2nd tensor to fit 1st tensor
        final_output = self.layer_dense(cov_euklidean)
        # final_output = self.layer_softmax2(conv1)
        return final_output

    def get_config(self):
        return {
            "data_layers": self.data_layers,
            "cov_layers": self.cov_layers,
            "inner_channels": self.inner_channels,
            "n_channels_main": self.n_channels_main,
            "N_exp": self.N_exp
        }

    def compute_output_shape(self, input_shape):
        return tf.concat([input_shape[0], input_shape[2], input_shape[2]], 0)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


def cal_logeig(features):
    """
    Modeified LogEig layer based on literature

    """
    # features = tf.reduce_mean(features, axis=[1], keepdims=True)
    # features = tf.squeeze(features, [1])
    [s_f, v_f] = tf.linalg.eigh(features)
    s_f = tf.math.log(s_f)
    s_f = tf.linalg.diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 1, 3, 2]))        # shape (N, C2, C1, C1)
    # features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    cov_euklidean = tf.transpose(features_t, [0, 2, 3, 1])      # shape (N, C1, C1, C2)
    # cov_euklidean = features_t
    return cov_euklidean


def _cal_log_cov(features):
    """
    Original LogEig from literature

    """
    features = tf.squeeze(features, [1])
    [s_f, v_f] = tf.linalg.eigh(features)
    s_f = tf.math.log(s_f)
    s_f = tf.linalg.diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t


class baseline(tf.keras.layers.Layer):
    """
    baseline layers
    """

    def __init__(self, n_channels=256):
        super(baseline, self).__init__()
        self.s1 = n_channels
        self.s2 = n_channels // 2

    def build(self, input_shape):
        self.w0 = self.add_weight(
            shape=(self.s1, self.s2),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
            dtype=self.dtype,
            name='orth_weight0'
        )

    def call(self, inputs):
        shape = tf.shape(inputs)
        reshaped = tf.reshape(inputs, [shape[0], shape[1] * shape[2], shape[3]])
        # reshaped = tf.reshape(inputs, shape=(-1, tf.shape(inputs)[1] * tf.shape(inputs)[2], tf.shape(inputs)[3]))
        # Cov Pooling Layer
        local5 = self._cal_cov_pooling(reshaped)
        # print('Name {}'.format(local5.shape))
        shape = tf.shape(local5)
        # BiRe Layer - 1
        weight1, weight2 = self._variable_with_orth_weight_decay('orth_weight0', shape)
        local6 = tf.matmul(tf.matmul(weight2, local5), weight1, name='matmulout')
        local7 = self._cal_rect_cov(local6)
        '''
        # Additional BiRe Layer
        shape = local7.get_shape().as_list()
        print('spdpooling feature2: D1:%d, D2:%d, D3:%d', shape[0], shape[1], shape[2])
        weight1, weight2 = _variable_with_orth_weight_decay('orth_weight1', shape)
        local8 = tf.matmul(tf.matmul(weight2, local7), weight1)
        '''
        local9 = self._cal_log_cov(local7)
        shape = tf.shape(local9)
        cov4 = tf.reshape(local9, [shape[0], shape[1] * shape[2]])

        return cov4

    def _cal_cov_pooling(self, features):
        shape_f = tf.shape(features)
        # shape_f[0] = -1
        centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]), 2)
        centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
        centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
        tmp = tf.subtract(features, centers_batch)
        tmp_t = tf.transpose(tmp, [0, 2, 1])
        features_t = 1 / tf.cast((shape_f[1] - 1), tf.float32) * tf.matmul(tmp_t, tmp)
        trace_t = tf.linalg.trace(features_t)
        trace_t = tf.reshape(trace_t, [shape_f[0], 1])
        trace_t = tf.tile(trace_t, [1, shape_f[2]])
        trace_t = 0.0001 * tf.linalg.diag(trace_t)
        return tf.add(features_t, trace_t)

    # Implementation is of basically LogEig Layer
    def _cal_log_cov(self, features):
        [s_f, v_f] = tf.linalg.eigh(features)
        s_f = tf.math.log(s_f)
        s_f = tf.linalg.diag(s_f)
        features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
        return features_t

    # computes weights for BiMap Layer
    def _variable_with_orth_weight_decay(self, name1, shape):
        self.s1 = tf.cast(shape[2], tf.int32)
        self.s2 = tf.cast(shape[2] / 2, tf.int32)
        # w0_init, _ = tf.linalg.qr(tf.random.normal([s1, s2], mean=0.0, stddev=1.0))
        # w0 = tf.Variable(initial_value=w0_init, name=name1)
        tmp1 = tf.reshape(self.w0, (1, self.s1, self.s2))
        tmp2 = tf.reshape(tf.transpose(self.w0), (1, self.s2, self.s1))
        if shape[0] is not None:
            tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
            tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
        return tmp1, tmp2

    # ReEig Layer
    def _cal_rect_cov(self, features):
        [s_f, v_f] = tf.linalg.eigh(features)
        s_f = tf.clip_by_value(s_f, 0.0001, 10000)
        s_f = tf.linalg.diag(s_f)
        features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
        return features_t


class layer_dense(tf.keras.layers.Layer):  # output the classes
    def __init__(self, num_classes=7):
        super(layer_dense, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = []

        self.dense_layers.append(tf.keras.layers.Dense(2000, activation=None, name='fc_1'))
        self.dense_layers.append(tf.keras.layers.Activation('relu'))
        self.dense_layers.append(tf.keras.layers.Dense(128, activation=None, name='fc_2'))
        self.dense_layers.append(tf.keras.layers.Activation('relu'))
        self.dense_layers.append(tf.keras.layers.Dense(num_classes, activation='softmax', name='Bottleneck'))

        # combine the dense layers together
        self.model = tf.keras.Sequential(self.dense_layers)

    def call(self, inputs):
        inputs_flatten = self.flatten(inputs)
        # inputs_flatten = tf.reshape(inputs, shape=(-1, tf.shape(inputs)[1] * tf.shape(inputs)[2] * tf.shape(inputs)[3]))
        conv = self.model(inputs_flatten)
        return conv


def feature_fusion(tensor1, tensor2, weight1=1, weight2=1):
    """
    This function is aimed to fusion the first and second dimension features...
    combine two tensors with different shapes (N, k, k, C) and (N, C, C, 1)
    """
    # feature_pooled1 = tf.reduce_mean(tensor1, axis=[2, 3], keepdims=True)   # shape (N, C, 1, 1)
    # feature_pooled2 = tf.reduce_mean(tensor2, axis=[2, 3], keepdims=True)   # shape (N, C2, 1, 1)
    # tensor1_t = tf.transpose(tensor1, [0, 2, 3, 1])  # shape (N, k, k, C)
    # tensor2_t = tf.transpose(tensor2, [0, 2, 3, 1])  # shape (N, C1, C1, C2)
    # shape1 = tensor1_t.shape
    shape1 = tf.shape(tensor1)
    tensor2 = tf.reduce_mean(tensor2, axis=[3], keepdims=True)  # shape (N, k, k, 1)
    tensor2_resize = tf.image.resize(tensor2, (shape1[1], shape1[2]))  # shape (N, C, C, 1)
    feature_combined = tf.concat([tensor1 * weight1, tensor2_resize * weight2], axis=-1)  # shape (N, C, C, 2)
    # feature_combined_t = tf.transpose(feature_combined, [0, 3, 1, 2])
    return feature_combined


def data_N_M_d_c_to_cov_N_C2_C1_C1_image(input, N_heads=5):
    """
    Convert input tensors of shape (N, M, d ,C) to covariance matrices of shape (N, C2, C1, C1).
    (Reference GitHub: https://github.com/d-acharya/CovPoolFER/blob/master/conv_feature_pooling/src/models/covpoolnet.py)

    Args:
        input: (tf.Tensor): Input tensor of shape (N, M, d, C).

    Returns:
        tf.Tensor: Covariance matrices of shape (N, C2, C1, C1).

    """
    # first flatten to (N,M*d,C)
    features = tf.reshape(input,
                          shape=(-1, tf.shape(input)[1] * tf.shape(input)[2], tf.shape(input)[3] // N_heads,
                                 N_heads))  # shape (N, M*d, C1, C2)
    features = tf.transpose(features, [0, 3, 2, 1])  # shape (N, C2, C1, M*d)
    # shape_f = features.get_shape().as_list()
    shape_f = tf.shape(features)
    # centers_batch = tf.reduce_mean(features, 1)     # shape (N, 1, C)
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 1, 3, 2]), 3)
    centers_batch = tf.reshape(centers_batch, [shape_f[0], shape_f[1], 1, shape_f[3]])  # shape (N, C2, 1, M*d)
    # centers_batch = tf.expand_dims(centers_batch, axis=1)
    centers_batch = tf.tile(centers_batch,
                            [1, 1, shape_f[2], 1])  # copy center batch to each feature/pixel shape (N, C2, C1, M*d)
    # compute the covariance matrices with the shape of (N, C2, C1, C1)
    tmp = tf.subtract(features, centers_batch)  # shape (N, C2, C1, M*d)
    tmp_t = tf.transpose(tmp, [0, 1, 3, 2])  # shape (N, C2, M*d, C1)
    cov = 1 / tf.cast((shape_f[1] - 1), tf.float32) * tf.matmul(tmp, tmp_t)  # shape (N, C2, C1, C1)
    # Add trace on the cov to ensures the positive definite and prevents numerical instability
    trace_t = tf.linalg.trace(cov)
    trace_t = tf.reshape(trace_t, [shape_f[0], shape_f[1], 1])
    trace_t = tf.tile(trace_t, [1, 1, shape_f[2]])
    trace_t = 0.0001 * tf.linalg.diag(trace_t)  # multiply small weight 0.0001
    cov_trace = tf.add(cov, trace_t)  # shape (N, C2, C1, C1)
    return cov_trace


def _cal_cov_pooling(input):
    """
    Original cov pooling from literature

    """
    features = tf.reshape(input, shape=(-1, tf.shape(input)[1] * tf.shape(input)[2], tf.shape(input)[3]))
    shape_f = tf.shape(features)
    # shape_f[0] = -1
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.linalg.trace(features_t)
    trace_t = tf.reshape(trace_t, [shape_f[0], 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.0001*tf.linalg.diag(trace_t)
    cov = tf.add(features_t, trace_t)  # shape (N, c, c)
    cov_expand =  tf.reshape(cov, shape=(-1, 1, tf.shape(cov)[1], tf.shape(cov)[2]))    # shape (N, 1, c, c)
    return cov_expand


class layer_N_M_d_1_to_N_x_x_C_conv(tf.keras.layers.Layer):  # reduce the complexity of img
    def __init__(self, out_filters=256):
        super(layer_N_M_d_1_to_N_x_x_C_conv, self).__init__()
        self.conv_layers = []
        # define the CNN architecture
        # 1
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))
        self.conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # 4
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))
        self.conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # 7
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))
        # 9
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))
        self.conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        # 12
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))
        # 14
        self.conv_layers.append(
            tf.keras.layers.Conv2D(filters=out_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation=None))
        self.conv_layers.append(tf.keras.layers.Activation('relu'))

        # combine the conv layers together
        self.model = tf.keras.Sequential(self.conv_layers)

    def call(self, inputs):
        return self.model(inputs)


class layer_softmax2(tf.keras.layers.Layer):
    """
    Calculates probabilities for the number of output classes
    """

    def __init__(self, num_classes=7, **kwargs):
        super(layer_softmax2, self).__init__(**kwargs)
        # self.ln_em = tf.keras.layers.LayerNormalization()
        self.num_classes = num_classes

    def build(self, input_shape):
        neurons_in = tf.cast(input_shape[3], tf.int32)
        self.w = self.add_weight(
            shape=(neurons_in, self.num_classes),  # the output shape
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            # constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            name='w'
        )

    def call(self, inputs, **kwargs):
        cov3 = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        cov3 = tf.squeeze(cov3, [1, 2])
        probs = tf.nn.softmax(tf.matmul(cov3, self.w), axis=1)
        return probs

    def get_config(self):
        return {}

    def compute_output_shape(self, input_shape):
        # return tf.concat([input_shape[0], 1, 1, 7], 0)        # the output shape
        return tf.concat([input_shape[0], input_shape[2], input_shape[3], self.num_classes], 0)  # the output shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class layer_N_M_d_1_to_N_M_d_C_residual(tf.keras.layers.Layer):
    """
        Embedding of [M,d] matrices into [M,d,C] tensors.

        Args:
            units_output (int): Number of channels.
            activation (str): Activation function to be applied.
    """

    def __init__(self, units_output, activation="relu", **kwargs):
        self.activation_string = activation
        self.units_output = units_output
        self.activation = tf.keras.activations.get(activation)
        self.units = int(self.units_output)
        self.input_units = 1
        super().__init__(**kwargs)
        self.w = self.add_weight(
            shape=(self.input_units, self.units_output),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"w"
        )
        self.w2 = self.add_weight(
            shape=(self.units_output, self.units_output),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"w2"
        )
        self.llambda = self.add_weight(
            shape=(self.units_output,),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"llambda"
        )

    def call(self, inputs, **kwargs):
        # Compute the output
        res = inputs + tf.multiply(
            tf.matmul(self.activation(tf.matmul(inputs, self.w)), self.w2), self.llambda)

        # Stack the outputs with M_matrix and d_matrix

        return res

    def get_config(self):
        return {
            "units_output": self.units_output,
            "activation": self.activation_string
        }

    def compute_output_shape(self, input_shape):
        return tf.concat([input_shape[0:2], tf.expand_dims(tf.cast(self.units_output, tf.int32), 0)], 0) + 2

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class layer_N_C_d_d_bilinear_attention_cov2cor_spd(tf.keras.layers.Layer):
    """
        A layer that first applies correlation normalization and then applies bilinear attention mechanism.
    """
    def __init__(self, **kwargs):
        super(layer_N_C_d_d_bilinear_attention_cov2cor_spd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_units = int(input_shape[1])
        def init2(shape,dtype=ddtype):
            return tf.cast(l1_constraint_columns(tf.abs(tf.random.normal(shape,0,1/tf.sqrt(tf.cast(shape[0],tf.float32))))), dtype)

        self.w_keys = self.add_weight(
            shape=(self.input_units, self.input_units),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=2/ tf.cast(self.input_units, tf.float32)),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            dtype=ddtype,
            name="w_keys"
        )
        self.w_queries = self.add_weight(
            shape=(self.input_units, self.input_units),
            initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=2/ tf.cast(self.input_units, tf.float32)),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            dtype=ddtype,
            name="w_queries"
        )

        self.w_values = self.add_weight(
            shape=(self.input_units, self.input_units),
            initializer=init2,
            #constraint=tf.keras.constraints.NonNeg(),
            constraint=l1_constraint_columns,
            trainable=True,
            dtype=ddtype,
            name="w_values"
        )


    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):
        dia = tf.abs(tf.linalg.diag_part(inputs))
        dia = tf.maximum(dia, 0.0001)
        di = tf.sqrt(dia)
        d_inv = tf.linalg.diag(1 / di)
        cor = tf.matmul(tf.matmul(d_inv, inputs), d_inv)

        cor = tf.minimum(cor, 1)
        cor = tf.maximum(cor, -1)
        X=cor

        inputs_cov2cor = tf.transpose(X, perm=[0, 2, 3, 1])
        XWQ = tf.matmul(inputs_cov2cor, self.w_queries)
        XWK = tf.matmul(inputs_cov2cor, self.w_keys)
        XWV = tf.matmul(inputs_cov2cor, self.w_values)
        XWQt = tf.transpose(XWQ, perm=[0, 3, 1, 2])
        XWKt = tf.transpose(XWK, perm=[0, 3, 2, 1])
        XWKT = tf.transpose(XWK, perm=[0, 3, 1, 2])
        #TRANSPOSE HERE FOR MORE STABILITY
        A_pre=tf.matmul(tf.matmul(XWKt, XWQt), XWKT)
        A= SoftPDmax_additiveScale_N_c_d_d(A_pre)
        AXA = tf.matmul(tf.matmul(A, tf.transpose(XWV,(0,3,1,2))), A,transpose_b=True)
        res = AXA
        res = inputs + res
        return res

    def get_config(self):
        return {}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class layer_N_C_d_d_spd_activation_scaled(tf.keras.layers.Layer):
    """
    A layer that applies SPD activation function to the input tensors.

    Args:
        N_exp (int): Number of exponentiation terms in the activation function.
    """
    def __init__(self, N_exp=3, **kwargs):
        self.N_exp = N_exp
        super(layer_N_C_d_d_spd_activation_scaled, self).__init__(**kwargs)

    def build(self, input_shape):
        self.C = int(input_shape[1])

        def custom_initializer(shape, dtype=None):
            # Create a tensor of zeros with the given shape
            init_tensor = tf.zeros(shape, dtype=dtype)
            # Set the first element to 1
            init_tensor = tf.tensor_scatter_nd_update(init_tensor, [[0]], [1.0])
            return init_tensor

        self.channel_weights = [
            self.add_weight(
                name=f'channel_weight_{c}',
                shape=(self.N_exp,),
                initializer=custom_initializer,  # Choose your initializer
                trainable=True,
                dtype=ddtype,
                constraint=l1_constraintLessEqual2
            ) for c in range(self.C)
        ]

    def call(self, inputs, **kwargs):
        # Initialize an empty tensor to store the results for each channel
        channel_results = []

        for c in range(self.C):
            channel_input = inputs[:, c, :, :]  # Select the channel
            channel_weight = self.channel_weights[c]

            dia = tf.abs(tf.linalg.diag_part(channel_input))
            dia = tf.maximum(dia, tf.expand_dims(tf.reduce_sum(dia, axis=1), axis=1) / 100000)
            dia = tf.maximum(dia, 0.0001)
            di = tf.sqrt(dia)
            diagonal_part = tf.linalg.diag(di)
            d_inv = tf.linalg.diag(1 / di)
            cors = tf.matmul(tf.matmul(d_inv, channel_input), d_inv)
            cors = tf.math.minimum(cors, 1)
            cors = tf.math.maximum(cors, -1)

            def body(i, res_exp_powers, summ):
                res_exp_powers *= cors
                summ += (channel_weight[i] * res_exp_powers)  # Use channel-specific weights
                return i + 1, res_exp_powers, summ

            i = tf.constant(1)
            res_exp_powers = cors
            summ = res_exp_powers * channel_weight[0]  # Use the first weight for initialization

            def while_cond(i, res_exp_powers, summ):
                return i < self.N_exp

            _, _, res = tf.while_loop(
                while_cond,
                body,
                [i, res_exp_powers, summ]
            )

            res = res + cors  # Add back the original correlation matrix
            channel_results.append(res)

        # Concatenate the results for each channel back together
        final_output = tf.stack(channel_results, axis=1)
        return final_output

    def get_config(self):
        return {"N_exp": self.N_exp}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def SoftPDmax_additiveScale_N_c_d_d(cov_xx):
    '''Custom Softmax function'''
    cov_xx = cov_xx - tf.reduce_max(cov_xx, axis=(2, 3), keepdims=True)  # just for computational stability
    cov_xx = tf.exp(cov_xx)
    dia = tf.reduce_sum(cov_xx, -1)
    dia = tf.maximum(dia, tf.expand_dims(tf.reduce_sum(dia, axis=2), axis=2) / 100000)
    dia = tf.maximum(dia, 0.00001)
    di = tf.sqrt(dia)
    d_inv = tf.linalg.diag(1 / di)
    cor = tf.matmul(tf.matmul(d_inv, cov_xx), d_inv)
    return cor


# @tf.function(reduce_retracing=True)
def l1_constraintLessEqual(w):
    w_g0 = w * tf.keras.backend.cast(tf.keras.backend.greater_equal(w, 0), tf.keras.backend.floatx()) \
           * tf.keras.backend.cast(tf.keras.backend.less_equal(tf.keras.backend.sum(w), 1), tf.keras.backend.floatx())
    return w_g0

def l1_constraintLessEqual2(w):
    # Step 1: Set negative weights to 0
    w = w * tf.keras.backend.cast(tf.keras.backend.greater_equal(w, 0), tf.keras.backend.floatx())

    # Step 2: Scale weights if their sum exceeds 1
    sum_w = tf.keras.backend.sum(w)
    w = tf.cond(sum_w > 1, lambda: w / sum_w, lambda: w)
    return w

# @tf.function(reduce_retracing=True)
def l1_constraint_columns(w):
    w_g0 = tf.math.maximum(w, 0)
    w_g0 = w_g0 / tf.math.reduce_sum(w_g0, axis=0)
    return w_g0

class MultiHeadAttention_N_C_d_d_bilinear(tf.keras.Model):
    """
        Multi-head attention layer for bilinear attention.
        Initializes and calls bilinear attention on several heads

        Args:
            num_heads (int): Number of attention heads.
            inner_channels (int): Number of inner channels.
            activation (str): Activation function. Default is "relu".
            **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttention_N_C_d_d_bilinear, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.head_units = int(input_shape[1] / self.num_heads)
        for l in range(1, self.num_heads + 1):
            setattr(self, f"head_{l}",
                    layer_N_C_d_d_bilinear_attention_cov2cor_spd())

    def call(self, inputs, **kwargs):
        attention_heads = tf.split(inputs, self.num_heads, axis=1)
        heads = [getattr(self, f"head_{l + 1}")(attention_heads[l]) for l in range(self.num_heads)]
        outputs = tf.concat(heads, axis=1)
        return outputs

    def get_config(self):
        return {'num_heads': self.num_heads}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)
