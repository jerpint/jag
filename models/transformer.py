import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import initializers as init
from tensorflow.keras import backend as K
import numpy as np
import math


class LayerNormalization(layers.Layer):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    dimension.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters.

    Args:
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, eps=1e-5, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=input_shape[-1:],
            initializer=init.Ones(),
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer=init.Zeros(),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Embedding(layers.Layer):
    r"""Implements the unified Embedding Layer of the Transformer architecture.
    It performs positional embeddings, token embeddings and, eventually, segment
    embeddings.

    Args:
        output_dim: dimension of the embeddings. Default: 768.
        dropout: dropout rate to be applied on the embeddings. Default: 0.1.
        vocab_size: size of the vocalbulary. Default: 30000.
        max_len: maximum length of the input sequence. Default: 512.
        trainable_pos_embedding: whether or not to train the positional embeddings.
    Default: ``True``.
        num_segments: number of segments. if None or set to zero, then the segment
    embeddings  won't be performed. Default: None.
        use_one_dropout: if ``True``, the different embeddings will be summed up
    before applying dropout, otherwise dropout will be applied to each embedding type
    independently before summing them. Default: ``False``.
        use_embedding_layer_norm: if ``True``, layer normalization will be applied on
    the resulting embeddings. Default: ``False``.
        layer_norm_epsilon: parameter of the layer normalization operation. Default: 1e-5

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L, output_dim)`
    """

    def __init__(
            self, output_dim=768, dropout=0.1, vocab_size=30000,
            max_len=512, trainable_pos_embedding=True, num_segments=None,
            use_one_dropout=False, use_embedding_layer_norm=False,
            layer_norm_epsilon=1e-5, **kwargs):

        super(Embedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.max_len = max_len
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.num_segments = num_segments
        self.trainable_pos_embedding = trainable_pos_embedding
        self.use_one_dropout = use_one_dropout
        self.use_embedding_layer_norm = use_embedding_layer_norm
        self.layer_norm_epsilon = layer_norm_epsilon

        if self.num_segments is None or (self.num_segments == 0):
            self.segment_emb = None
        else:
            self.segment_emb = layers.Embedding(
                self.num_segments, self.output_dim,
                input_length=self.max_len
            )

        if self.trainable_pos_embedding:
            self.pos_emb = layers.Embedding(
                max_len, output_dim, input_length=max_len
            )
        else:
            self.pos_emb = layers.Embedding(
                max_len, output_dim, input_length=max_len,
                trainable=False,
                weights=[Embedding._get_pos_encoding_matrix(
                    max_len, output_dim)]
                # embeddings_initializer=Embedding._get_pos_encoding_matrix(
                #     max_len, output_dim
                # )
            )

        self.token_emb = layers.Embedding(
            self.vocab_size, output_dim, input_length=max_len
        )

        self.embedding_dropout = layers.Dropout(self.dropout)
        self.add_embeddings = layers.Add()

        if self.use_embedding_layer_norm:
            self.embedding_layer_norm = LayerNormalization(
                self.layer_norm_epsilon)
        else:
            self.embedding_layer_norm = None

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[0]).as_list()
        shape.append(self.output_dim)
        return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'max_len': self.max_len,
            'use_one_dropout': self.use_one_dropout,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'num_segments': self.num_segments,
            'vocab_size': self.vocab_size,
            'trainable_pos_embedding': self.trainable_pos_embedding,
            'use_embedding_layer_norm': self.use_embedding_layer_norm,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        if self.num_segments is None or (self.num_segments == 0):
            tokens, pos_ids = inputs
            segment_embedding = None
        else:
            tokens, segment_ids, pos_ids = inputs
            segment_embedding = self.segment_emb(segment_ids)

        pos_embedding = self.pos_emb(pos_ids)
        token_embedding = self.token_emb(tokens)

        if self.use_one_dropout:
            embed_list = [] if segment_embedding is None else [segment_embedding]
            embed_list.extend([pos_embedding, token_embedding])
            sum_embed = self.add_embeddings(embed_list)
            if self.embedding_layer_norm is not None:
                sum_embed = self.embedding_layer_norm(sum_embed)

            return self.embedding_dropout(sum_embed)

        else:
            embed_list = [] if segment_embedding is None else [
                self.embedding_dropout(segment_embedding)
            ]
            embed_list.extend([
                self.embedding_dropout(pos_embedding),
                self.embedding_dropout(token_embedding)
            ])
            sum_embed = self.add_embeddings(embed_list)

            if self.embedding_layer_norm is not None:
                sum_embed = self.embedding_layer_norm(sum_embed)

            return sum_embed

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def _get_pos_encoding_matrix(max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb)
                 for j in range(d_emb)]
                if pos != 0 else np.zeros(d_emb)
                for pos in range(max_len)
            ],
            dtype=np.float32
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc


class ScaledDotProductAttention(layers.Layer):
    r"""Implements the scaled dot product attention mechanism.

    Args:
        temperature: the normalizing constant.
        attn_dropout: dropout rate to be applied on the result. Default: 0.1.
        use_attn_mask: whether or not the layer expects to use mask in the computation.
    Default: ``False``.
        neg_inf: constant representing the negative infinite value. Default: ``-np.inf``.

    Inputs:
        ``query``: the query of dimension :math:`(N, H, L, Dk)`
        ``keys``: the keys of dimension :math:`(N, H, Dk, L)`
        ``values``: the values of dimension :math:`(N, H, L, Dv)`
        ``mask`` (only if use_attn_mask is True): the mask of dimension
    :math:`(N, 1, L, L)`.
    Outputs:
        ``result``: the result of the operation :math:`(N, H, L, Dv)`
        ``attention_weight``: the attention values :math:`(N, H, L, L)`
    """

    def __init__(
            self, temperature, attn_dropout=0.1, use_attn_mask=False,
            neg_inf=-np.inf, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.temperature = temperature
        self.attn_dropout = attn_dropout
        self.neg_inf = neg_inf
        self.use_attn_mask = use_attn_mask

        self.dropout = layers.Dropout(self.attn_dropout)
        self.softmax = layers.Softmax(axis=-1)

    def compute_output_shape(self, input_shape):
        shape1 = tf.TensorShape(input_shape[0]).as_list()
        shape2 = tf.TensorShape(input_shape[1]).as_list()
        shape3 = tf.TensorShape(input_shape[2]).as_list()

        ret_shape1 = list(shape1)
        ret_shape1[-1] = shape3[-1]

        ret_shape2 = list(shape1)
        ret_shape2[-1] = shape2[-1]

        ret_shape1 = tf.TensorShape(ret_shape1)
        ret_shape2 = tf.TensorShape(ret_shape2)

        return [ret_shape1, ret_shape2]

    def call(self, inputs):
        # q and v are B, H, L, C//H ; k is B, H, C//H, L ; mask is B, 1, L, L
        # q: B, H, lq, dk and v: B, H, lv, dv and k:B, H, dk, Lk and mask: B,
        # 1, Lq, Lk
        if self.use_attn_mask:
            q, k, v, mask = inputs
        else:
            q, k, v = inputs
            mask = None

        attn = K.batch_dot(q, k)  # attn is of shape B, H, Lq, Lk
        attn = attn / self.temperature
        if mask is not None:
            attn = mask * attn + (1.0 - mask) * self.neg_inf
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = K.batch_dot(attn, v)  # output: B, H, L, C//H (B, H, Lq, dv)

        return [output, attn]

    def get_config(self):
        config = {
            'temperature': self.temperature,
            'attn_dropout': self.attn_dropout,
            'neg_inf': self.neg_inf,
            'use_attn_mask': self.use_attn_mask,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadAttention(layers.Layer):
    r"""Implements the multi head attention mechanism.

    Args:
        n_head: number of heads.
        d_model: dimension of the ouput results.
        d_k: dimension of the keys and the queries.
        d_v: dimension of the values.
        attention_dropout: dropout rate to be applied on each single attention
    head. Default: 0.1.
        dropout: dropout rate to be applied on the projection of
    the concatenation of all attention heads. Default: 0.1.
        use_attn_mask: whether or not the layer expects to use mask in the computation.
    Default: ``False``.
        layer_norm_epsilon: parameter of the layer normalization operation. Default: 1e-5
        neg_inf: constant representing the negative infinite value. Default: ``-np.inf``.

    Inputs:
        ``seq``: the input sequence of dimension :math:`(N, L, d_model)`
        ``mask`` (only if use_attn_mask is True): the mask of dimension
    :math:`(N, 1, L, L)`.
    Outputs:
        ``result``: the result of the operation :math:`(N, L, d_model)`
        ``attention_weight``: the attention values :math:`(N, n_head, L, L)`
    """

    def __init__(
            self, n_head, d_model, d_k, d_v, attention_dropout=0.1,
            dropout=0.1, use_attn_mask=False, layer_norm_epsilon=1e-5,
            neg_inf=-np.inf, **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention_dropout = attention_dropout
        self.use_attn_mask = use_attn_mask
        self.neg_inf = neg_inf
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.c_attn = layers.Conv1D(
            n_head * (d_k * 2 + d_v), 1, input_shape=(None, self.d_model)
        )
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5),
            attn_dropout=self.attention_dropout,
            neg_inf=self.neg_inf,
            use_attn_mask=self.use_attn_mask
        )
        self.c_attn_proj = layers.Conv1D(
            d_model, 1, input_shape=(None, n_head * d_v)
        )
        self.multihead_dropout = layers.Dropout(self.dropout)
        self.multihead_add = layers.Add()
        self.multihead_norm = LayerNormalization(self.layer_norm_epsilon)

    @staticmethod
    def _shape_list(x):
        tmp = K.int_shape(x)
        tmp = list(tmp)
        tmp[0] = -1
        return tmp

    @staticmethod
    def _split_heads(x, n, k=False):
        x_shape = MultiHeadAttention._shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1] + [n, m // n]
        new_x = K.reshape(x, new_x_shape)
        return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])

    @staticmethod
    def _merge_heads(x):
        new_x = K.permute_dimensions(x, [0, 2, 1, 3])
        x_shape = MultiHeadAttention._shape_list(new_x)
        new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
        return K.reshape(new_x, new_x_shape)

    def compute_output_shape(self, input_shape):
        x = input_shape[0] if self.use_attn_mask else input_shape
        shape = tf.TensorShape(x).as_list()

        shape1 = list(shape)
        shape1[-1] = self.d_model

        shape2 = [shape1[0], self.n_head, shape1[1], shape1[1]]

        ret_shape1 = tf.TensorShape(shape1)
        ret_shape2 = tf.TensorShape(shape2)

        return [ret_shape1, ret_shape2]

    def call(self, inputs):
        if self.use_attn_mask:
            x, mask = inputs
        else:
            x = inputs
            mask = None

        residual = x

        x = self.c_attn(x)

        q_l = self.n_head * self.d_k
        k_l = 2 * self.n_head * self.d_k
        q, k, v = x[:, :, :q_l], x[:, :, q_l:k_l], x[:, :, k_l:]

        q = MultiHeadAttention._split_heads(q, self.n_head)  # B, H, L, d_k
        k = MultiHeadAttention._split_heads(
            k, self.n_head, k=True)  # B, H, d_k, L
        v = MultiHeadAttention._split_heads(v, self.n_head)  # B, H, L, d_v

        args = [q, k, v]
        if self.use_attn_mask:
            args.append(mask)
        output, attn = self.attention(args)  # (B, H, Lq, dv), (B, H, Lq, Lk)

        output = MultiHeadAttention._merge_heads(output)  # (B, Lq, H x dv)
        output = self.c_attn_proj(output)

        output = self.multihead_dropout(output)
        output = self.multihead_norm(self.multihead_add([output, residual]))

        return [output, attn]

    def get_config(self):
        config = {
            'n_head': self.n_head,
            'd_model': self.d_model,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'attention_dropout': self.attention_dropout,
            'use_attn_mask': self.use_attn_mask,
            'neg_inf': self.neg_inf,
            'dropout': self.dropout,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GeLU(layers.Layer):
    r"""Implementation of the gelu activation function as described in
    the paper `Gaussian Error Linear Units (GELUs)`_ .
    .. math::
        0.5 * x * (1 + tanh(\sqrt(2 / \pi) * (x + 0.044715 * pow(x, 3))))

    Args:
        accurate: if ``False``, an approximate of this function is computed.
    Default: ``False``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    .. _`Gaussian Error Linear Units (GELUs)`: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, accurate=False, **kwargs):
        super().__init__(**kwargs)
        self.accurate = accurate

    def call(self, x, **kwargs):
        if not self.accurate:
            ouput = 0.5 * x * (
                1.0 + K.tanh(math.sqrt(2 / math.pi) *
                             (x + 0.044715 * K.pow(x, 3)))
            )
            return ouput
        else:
            return x * 0.5 * (1.0 + tf.erf(x / math.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'accurate': self.accurate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionwiseFeedForward(layers.Layer):
    r"""Implements the position wise feed forward network.

    Args:
        d_in: dimension of the input data.
        d_hid: dimension of the intermediate dense layer.
        dropout: dropout rate to be applied on the results. Default: 0.1.
        d_out: dimension of the output data. if ``None``, it is set to d_in.
        layer_norm_epsilon: parameter of the layer normalization operation. Default: 1e-5
        use_gelu: if ``True``, use the ``GeLU`` activation layer instead of
    the ``ReLU`` one.  Default: ``False``
        accurate_gelu: whether or not to use accurate (vs approximate)
    computation of the `GeLU`` operator. Default: ``False``.

    Shape:
        - Input: :math:`(N, L, d_in)`
        - Output: :math:`(N, L, d_out)`
    """

    def __init__(
            self, d_in, d_hid, dropout=0.1, d_out=None,
            layer_norm_epsilon=1e-5, use_gelu=False,
            accurate_gelu=False, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu

        self.conv1 = layers.Conv1D(
            self.d_hid, 1, input_shape=(None, self.d_in)
        )
        self.conv2 = layers.Conv1D(
            self.d_out, 1, input_shape=(None, self.d_hid)
        )

        if not self.use_gelu:
            self.activation = layers.ReLU()
        else:
            self.activation = GeLU(accurate=self.accurate_gelu)

        self.pff_dropout = layers.Dropout(self.dropout)
        self.pff_add = layers.Add()
        self.pff_norm = LayerNormalization(self.layer_norm_epsilon)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.d_out
        return tf.TensorShape(shape)

    def call(self, x):
        residual = x
        output = self.conv2(self.activation(self.conv1(x)))

        output = self.pff_dropout(output)

        if (self.d_out == self.d_in):
            output = self.pff_norm(self.pff_add([output, residual]))
        elif (self.d_out % self.d_in == 0):
            tmp = K.int_shape(residual)

            tmp1 = list(tmp)
            tmp1.append(self.d_out // self.d_in)
            new_o = K.reshape(output, tmp1)

            tmp2 = list(tmp)
            tmp2.append(1)
            new_r = K.reshape(residual, tmp2)

            output = self.pff_add([new_o, new_r])
            tmp3 = list(tmp)
            tmp3[-1] = self.d_out

            output = K.reshape(output, tmp3)
            output = self.pff_norm(output)

        else:
            output = self.pff_norm(output)

        return output

    def get_config(self):
        config = {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'd_hid': self.d_hid,
            'dropout': self.dropout,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Pooler(layers.Layer):
    """ Implements the pooling operation of the transformer architecture.
    This is done by simply taking the hidden state corresponding to the first token
    on which some nonlinear transformations are performed.

    Args:
        d_hid: dimension of the input data.

    Shape:
        - Input: :math:`(N, L, d_hid)`
        - Output: :math:`(N, d_hid)`

    """

    def __init__(self, d_hid, **kwargs):
        super(Pooler, self).__init__(**kwargs)

        self.d_hid = d_hid

        self.dense = layers.Dense(
            self.d_hid, input_shape=(self.d_hid, )
        )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-2] = self.d_hid
        return tf.TensorShape(shape[:-1])

    def call(self, x):
        first_token = x[:, 0]
        pooled_output = K.tanh(self.dense(first_token))
        return pooled_output

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LMPredictionHeadTransform(layers.Layer):

    def __init__(
            self, d_hid, layer_norm_epsilon=1e-5,
            use_gelu=True, accurate_gelu=False, **kwargs):
        super(LMPredictionHeadTransform, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu
        self.layer_norm_epsilon = layer_norm_epsilon

        self.layerNorm = LayerNormalization(self.layer_norm_epsilon)

        self.dense = layers.Dense(
            self.d_hid, input_shape=(self.d_hid, )
        )

        if not self.use_gelu:
            self.activation = layers.ReLU()
        else:
            self.activation = GeLU(accurate=self.accurate_gelu)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.d_hid
        return tf.TensorShape(shape)

    def call(self, x):
        return self.layerNorm(self.activation(self.dense(x)))

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LMPredictionHead(layers.Layer):
    """Implements a module for handling Masked Language Modeling task.

    """

    def __init__(
            self, d_hid, vocab_size, embedding_weights,
            layer_norm_epsilon=1e-5, use_gelu=True,
            accurate_gelu=False, **kwargs):
        super(LMPredictionHead, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu
        self.layer_norm_epsilon = layer_norm_epsilon
        self.vocab_size = vocab_size

        output_shape = [vocab_size]
        output_shape = tf.TensorShape(output_shape)

        self.output_bias = self.add_weight(
            name='output_bias',
            shape=output_shape,
            initializer=init.Zeros(),
            trainable=True
        )

        self.transform = LMPredictionHeadTransform(
            d_hid, layer_norm_epsilon, use_gelu, accurate_gelu
        )

        # self.decoder = layers.Dense(
        #     self.vocab_size, use_bias=False
        # )
        # self.decoder.build([None, d_hid])
        # # self.decoder.set_weights([embedding_weights])
        # self.decoder.kernel = embedding_weights

        self.decoder = embedding_weights

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.vocab_size
        return tf.TensorShape(shape)

    def call(self, x):
        x = self.transform(x)
        # x = self.decoder(x)
        x = K.dot(x, K.transpose(self.decoder.embeddings))
        x = x + self.output_bias
        return x

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'vocab_size': self.vocab_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskedLM_NextSentenceHead(layers.Layer):
    """Implements a module for handling both Masked Language Modeling task
    and Next sentence prediction task.

    """

    def __init__(
            self, d_hid, vocab_size, embedding_weights=None,
            layer_norm_epsilon=1e-5, use_gelu=True,
            accurate_gelu=False, use_masked_lm=True,
            use_next_sp=True, **kwargs):
        super(MaskedLM_NextSentenceHead, self).__init__(**kwargs)
        assert (use_next_sp or use_masked_lm)
        self.d_hid = d_hid
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu
        self.layer_norm_epsilon = layer_norm_epsilon
        self.vocab_size = vocab_size
        self.use_masked_lm = use_masked_lm
        self.use_next_sp = use_next_sp

        if self.use_masked_lm:
            assert (embedding_weights is not None)
            self.predictions = LMPredictionHead(
                d_hid, vocab_size, embedding_weights,
                layer_norm_epsilon, use_gelu, accurate_gelu
            )

        if self.use_next_sp:
            self.seq_relationship = layers.Dense(2, input_shape=(self.d_hid, ))

    def compute_output_shape(self, input_shape):
        if self.use_masked_lm and self.use_next_sp:
            shape1 = tf.TensorShape(input_shape[0]).as_list()
            shape2 = tf.TensorShape(input_shape[1]).as_list()
            shape1[-1] = self.vocab_size
            shape2[-1] = 2
            return [tf.TensorShape(shape1), tf.TensorShape(shape2)]

        elif self.use_masked_lm:
            shape1 = tf.TensorShape(input_shape).as_list()
            shape1[-1] = self.vocab_size
            return tf.TensorShape(shape1)

        elif self.use_next_sp:
            shape1 = tf.TensorShape(input_shape).as_list()
            shape1[-1] = 2
            return tf.TensorShape(shape1)

        raise ValueError('incompatible mode')

    def call(self, inputs):
        if self.use_masked_lm and self.use_next_sp:
            sequence_output, pooled_output = inputs
        elif self.use_masked_lm:
            sequence_output = inputs
        elif self.use_next_sp:
            pooled_output = inputs

        output = []
        if self.use_masked_lm:
            a = self.predictions(sequence_output)
            output.append(a)

        if self.use_next_sp:
            b = self.seq_relationship(pooled_output)
            output.append(b)

        if len(output) == 1:
            output = output[0]

        return output

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'vocab_size': self.vocab_size,
            'use_next_sp': self.use_next_sp,
            'use_masked_lm': self.use_masked_lm,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceClassificationTask(layers.Layer):
    """Implements a module for handling sequence level classification task.

    """

    def __init__(
            self, d_hid, num_labels, dropout=0.1, **kwargs):
        super(SequenceClassificationTask, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.num_labels = num_labels
        self.dropout = dropout

        self.seq_class_dropout = layers.Dropout(self.dropout)
        self.classifier = layers.Dense(
            self.num_labels, input_shape=(self.d_hid, )
        )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_labels
        return tf.TensorShape(shape)

    def call(self, pooled):
        x = self.seq_class_dropout(pooled)
        x = self.classifier(x)
        return x

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'num_labels': self.num_labels,
            'dropout': self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultipleChoiceTask(layers.Layer):
    """Implements a module for handling multiple choice task.

    """

    def __init__(
            self, d_hid, num_choices, dropout=0.1, **kwargs):
        super(MultipleChoiceTask, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_choices = num_choices

        self.mod_dropout = layers.Dropout(self.dropout)
        self.classifier = layers.Dense(
            1, input_shape=(self.d_hid, )
        )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_labels
        return tf.TensorShape(shape)

    def call(self, pooled):
        x = self.mod_dropout(pooled)
        x = self.classifier(x)
        x = K.reshape(x, [-1, self.num_choices])
        return x

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'dropout': self.dropout,
            'num_choices': self.num_choices
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenClassificationTask(layers.Layer):
    """Implements a module for handling token level classification task.

    """

    def __init__(
            self, d_hid, num_labels, dropout=0.1, **kwargs):
        super(TokenClassificationTask, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.num_labels = num_labels
        self.dropout = dropout

        self.mod_dropout = layers.Dropout(self.dropout)
        self.classifier = layers.Dense(
            self.num_labels, input_shape=(self.d_hid, )
        )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_labels
        return tf.TensorShape(shape)

    def call(self, x):
        x = self.mod_dropout(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'num_labels': self.num_labels,
            'dropout': self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class QuestionAnsweringTask(layers.Layer):
    """Implements a module for handling token level classification task.

    """

    def __init__(
            self, d_hid, dropout=0.1, **kwargs):
        super(QuestionAnsweringTask, self).__init__(**kwargs)
        self.d_hid = d_hid
        self.dropout = dropout

        self.mod_dropout = layers.Dropout(self.dropout)
        self.qa_outputs = layers.Dense(
            2, input_shape=(self.d_hid, )
        )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return [tf.TensorShape(shape[:-2]), tf.TensorShape(shape[:-2])]

    def call(self, x):
        x = self.mod_dropout(x)
        x = self.qa_outputs(x)

        start_logits = x[:, :, 0]
        end_logits = x[:, :, 1]

        return [start_logits, end_logits]

    def get_config(self):
        config = {
            'd_hid': self.d_hid,
            'dropout': self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EncoderLayer(layers.Layer):
    r"""Implements an encoder layer of the transformer architecture.

    Args:
        d_model: dimension of the input data.
        d_inner: dimension of the intermediate hidden layer.
        n_head: number of heads.
        d_k: dimension of the keys and the queries.
        d_v: dimension of the values.
        d_out: dimension of the output data. if ``None``, it is set to d_model.
        residual_dropout: dropout rate to be applied on each residual operation
    results. Default: 0.1.
        attention_dropout: dropout rate to be applied on each attention
    mechanism results. Default: 0.1.
        use_pad_mask: whether or not the layer expects to use pad mask in the
    computation. Default: ``False``.
        use_attn_mask: whether or not the layer expects to use attention mask
    in the computation. Default: ``True``.
        neg_inf: constant representing the negative infinite value. Default: ``-np.inf``.
        ln_epsilon: parameter of the layer normalization operation. Default: 1e-5
        use_gelu: if ``True``, use the ``GeLU`` activation layer instead of
    the ``ReLU`` one.  Default: ``True``
        accurate_gelu: whether or not to use accurate (vs approximate)
    computation of the `GeLU`` operator. Default: ``False``.


    Inputs:
        ``seq``: the input sequence of dimension :math:`(N, L, d_model)`
        ``attn_mask`` (only if use_attn_mask is True): the attn_mask of dimension
    :math:`(N, 1, L, L)`.
        ``pad_mask`` (only if use_pad_mask is True): the pad_mask of dimension
    :math:`(N, L, 1)`.
    Outputs:
        ``result``: the result of the operation :math:`(N, L, d_out)`
        ``attention_weight``: the attention values :math:`(N, n_head, L, L)`
    """

    def __init__(
            self, d_model, d_inner, n_head, d_k, d_v, d_out=None,
            residual_dropout=0.1, attention_dropout=0.1, use_pad_mask=False,
            use_attn_mask=True, neg_inf=-np.inf, ln_epsilon=1e-5,
            use_gelu=True, accurate_gelu=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        if d_out is None:
            d_out = d_model
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.use_attn_mask = use_attn_mask
        self.use_pad_mask = use_pad_mask
        self.neg_inf = neg_inf
        self.ln_epsilon = ln_epsilon
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, attention_dropout=attention_dropout,
            dropout=residual_dropout, use_attn_mask=use_attn_mask,
            layer_norm_epsilon=ln_epsilon, neg_inf=neg_inf
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=residual_dropout, d_out=d_out,
            layer_norm_epsilon=ln_epsilon, use_gelu=use_gelu,
            accurate_gelu=accurate_gelu
        )

    def compute_output_shape(self, input_shape):
        if self.use_attn_mask or self.use_pad_mask:
            shape = tf.TensorShape(input_shape[0]).as_list()
        else:
            shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.d_out
        shape2 = [shape[0], self.n_head, shape[1], shape[1]]

        return [tf.TensorShape(shape), tf.TensorShape(shape2)]

    def call(self, inputs):
        if self.use_attn_mask and self.use_pad_mask:
            x, attn_mask, pad_mask = inputs
        elif self.use_attn_mask:
            x, attn_mask = inputs
            pad_mask = None
        elif self.use_pad_mask:
            x, pad_mask = inputs
            attn_mask = None
        else:
            x = inputs
            attn_mask = None
            pad_mask = None
        args = [x]
        if self.use_attn_mask:
            args.append(attn_mask)
        if len(args) == 1:
            args = args[0]
        output, attn = self.slf_attn(args)

        if self.use_pad_mask:
            output = pad_mask * output

        output = self.pos_ffn(output)
        if self.use_pad_mask:
            output = pad_mask * output

        return [output, attn]

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'n_head': self.n_head,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_out': self.d_out,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'use_attn_mask': self.use_attn_mask,
            'use_pad_mask': self.use_pad_mask,
            'neg_inf': self.neg_inf,
            'ln_epsilon': self.ln_epsilon,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DecoderLayer(layers.Layer):
    r"""Implements a decoder layer of the transformer architecture.

    Args:
        d_model: dimension of the input data.
        d_inner: dimension of the intermediate hidden layer.
        n_head: number of heads.
        d_k: dimension of the keys and the queries.
        d_v: dimension of the values.
        d_out: dimension of the output data. if ``None``, it is set to d_model.
        residual_dropout: dropout rate to be applied on each residual operation
    results. Default: 0.1.
        attention_dropout: dropout rate to be applied on each attention
    mechanism results. Default: 0.1.
        use_pad_mask: whether or not the layer expects to use pad mask in
    the computation. Default: ``False``.
        use_attn_mask: whether or not the layer expects to use attention
    mask in the computation. Default: ``True``.
        use_enc_output: whether or not the layer expects to use ouputs from
    the encoder in the computation. Default: ``True``.
        use_enc_mask: whether or not the layer expects to use the masks from the
    encoder in the computation. Default: ``False``.
        neg_inf: constant representing the negative infinite value. Default: ``-np.inf``.
        ln_epsilon: parameter of the layer normalization operation. Default: 1e-5
        use_gelu: if ``True``, use the ``GeLU`` activation layer instead of
    the ``ReLU`` one.  Default: ``True``
        accurate_gelu: whether or not to use accurate (vs approximate)
    computation of the `GeLU`` operator. Default: ``False``.


    Inputs:
        ``seq``: the input sequence of dimension :math:`(N, L, d_model)`
        ``enc_ouputs``(only if use_enc_output is True): the output of
    the encoder :math:`(N, Le, d_model)`
        ``attn_mask`` (only if use_attn_mask is True): the attn_mask of dimension
    :math:`(N, 1, L, L)`.
        ``pad_mask`` (only if use_pad_mask is True): the pad_mask of dimension
    :math:`(N, L, 1)`.
        ``enc_mask`` (only if use_enc_mask is True): the enc_mask of dimension
    :math:`(N, 1, Le, Le)`.
    Outputs:
        ``result``: the result of the operation :math:`(N, L, d_out)`
        ``attention_weight``: the attention values :math:`(N, n_head, L, L)`
        ``enc_attention_weight`` (only if use_enc_output is True): the attention
    values on encoder outputs :math:`(N, n_head, L, Le)`
    """

    def __init__(
            self, d_model, d_inner, n_head, d_k, d_v, d_out=None,
            residual_dropout=0.1, attention_dropout=0.1, use_pad_mask=False,
            use_attn_mask=True, use_enc_output=True, use_enc_mask=False, neg_inf=-np.inf,
            ln_epsilon=1e-5, use_gelu=True, accurate_gelu=False, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        if d_out is None:
            d_out = d_model
        if not use_enc_output:
            use_enc_mask = False
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.use_attn_mask = use_attn_mask
        self.use_pad_mask = use_pad_mask
        self.use_enc_mask = use_enc_mask
        self.use_enc_output = use_enc_output
        self.neg_inf = neg_inf
        self.ln_epsilon = ln_epsilon
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, attention_dropout=attention_dropout,
            dropout=residual_dropout, use_attn_mask=use_attn_mask,
            layer_norm_epsilon=ln_epsilon, neg_inf=neg_inf
        )
        if self.use_enc_output:
            self.enc_attn = MultiHeadAttention(
                n_head, d_model, d_k, d_v, attention_dropout=attention_dropout,
                dropout=residual_dropout, use_attn_mask=use_enc_mask,
                layer_norm_epsilon=ln_epsilon, neg_inf=neg_inf
            )
        else:
            self.enc_attn = None
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=residual_dropout, d_out=d_out,
            layer_norm_epsilon=ln_epsilon, use_gelu=use_gelu,
            accurate_gelu=accurate_gelu
        )

    def compute_output_shape(self, input_shape):
        if self.use_enc_output:
            shape = tf.TensorShape(input_shape[0]).as_list()
        else:
            shape = tf.TensorShape(input_shape).as_list()
        shape = tf.TensorShape(input_shape[0]).as_list()
        shape[-1] = self.d_out
        shape2 = [shape[0], self.n_head, shape[1], shape[1]]

        output_shape = [tf.TensorShape(shape), tf.TensorShape(shape2)]

        if self.use_enc_output:
            output_shape.append(tf.TensorShape(shape2))

        return output_shape

    def call(self, inputs):
        if self.use_attn_mask and self.use_pad_mask and self.use_enc_mask:
            x, enc, attn_mask, pad_mask, enc_mask = inputs
        elif self.use_attn_mask and self.use_pad_mask:
            if self.use_enc_output:
                x, enc, attn_mask, pad_mask = inputs
                enc_mask = None
            else:
                x, attn_mask, pad_mask = inputs
                enc = None
                enc_mask = None
        elif self.use_attn_mask and self.use_enc_mask:
            x, enc, attn_mask, enc_mask = inputs
            pad_mask = None
        elif self.use_pad_mask and self.use_enc_mask:
            x, enc, pad_mask, enc_mask = inputs
            attn_mask = None
        elif self.use_attn_mask:
            if self.use_enc_output:
                x, enc, attn_mask = inputs
                pad_mask = None
                enc_mask = None
            else:
                x, attn_mask = inputs
                pad_mask = None
                enc_mask = None
                enc = None
        elif self.use_pad_mask:
            if self.use_enc_output:
                x, enc, pad_mask = inputs
                attn_mask = None
                enc_mask = None
            else:
                x, pad_mask = inputs
                attn_mask = None
                enc_mask = None
                enc = None
        elif self.use_enc_mask:
            x, enc, enc_mask = inputs
            attn_mask = None
            pad_mask = None
        else:
            if self.use_enc_output:
                x, enc = inputs
            else:
                x = inputs
            enc_mask = None
            attn_mask = None
            pad_mask = None

        args = [x]
        if self.use_attn_mask:
            args.append(attn_mask)
        if len(args) == 1:
            args = args[0]
        output, attn = self.slf_attn(args)

        if self.use_pad_mask:
            output = pad_mask * output

        if self.use_enc_output:
            args = [output]
            if self.use_enc_mask:
                args.append(enc_mask)
            if len(args) == 1:
                args = args[0]
            output, dec_attn = self.enc_attn(args)

            if self.use_pad_mask:
                output = pad_mask * output
        else:
            dec_attn = None

        output = self.pos_ffn(output)
        if self.use_pad_mask:
            output = pad_mask * output

        return [output, attn] if dec_attn is None else [output, attn, dec_attn]

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'n_head': self.n_head,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_out': self.d_out,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'use_attn_mask': self.use_attn_mask,
            'use_pad_mask': self.use_pad_mask,
            'use_enc_mask': self.use_enc_mask,
            'use_enc_output': self.use_enc_output,
            'neg_inf': self.neg_inf,
            'ln_epsilon': self.ln_epsilon,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerEncoder(models.Model):
    r"""Implements an encoder layer of the transformer architecture.

    Args:
        vocab_size: size of the vocalbulary. Default: 30000.
        n_layers: number of layers. Default: 12.
        d_model: dimension of the embeddings. Default: 768.
        d_inner: dimension of the intermediate hidden layer. Default: 3072.
        n_head: number of heads. Default: 12.
        d_k: dimension of the keys and the queries. Default: 64.
        d_v: dimension of the values. Default: 64.
        d_out: dimension of the output data. if ``None``, it is set to d_model.
    Default: 768.
        max_len: maximum length of the input sequence. Default: 512.
        num_segments: number of segments. if None or set to zero, then the segment
    embeddings  won't be performed. Default: 2.
        embedding_dropout: dropout rate to be applied on embedding results. Default: 0.1.
        attention_dropout: dropout rate to be applied on each attention
    mechanism results. Default: 0.1.
        residual_dropout: dropout rate to be applied on each residual operation
    results. Default: 0.1.
        embedding_layer_norm: if ``True``, layer normalization will be applied on
    the resulting embeddings. Default: ``False``.
        layer_norm_epsilon: parameter of the layer normalization operation. Default: 1e-5
        neg_inf: constant representing the negative infinite value. Default: ``-1e9``.
        trainable_pos_embedding: whether or not to train the positional embeddings.
    Default: ``True``.
        use_one_embedding_dropout: if ``True``, the different embeddings will be
    summed up before applying dropout, otherwise dropout will be applied to each
    embedding type independently before summing them. Default: ``False``.
        use_attn_mask: whether or not the layer expects to use attention mask in the
    computation. Default: ``True``.
        use_pad_mask: whether or not the layer expects to use pad mask in the computation
    Default: ``False``.
        use_gelu: if ``True``, use the ``GeLU`` activation layer instead of
    the ``ReLU`` one.  Default: ``True``
        accurate_gelu: whether or not to use accurate (vs approximate)
    computation of the `GeLU`` operator. Default: ``False``.
        use_pooler: whether or not to compute the pooled representation of
    the input sequnces. Default: ``False``.
        use_masked_lm: whether or not to compute the masked language modeling outputs.
    Default: ``False``.
        use_next_sp: whether or not to compute the outputs of the next
    sentence prediction task. Default: ``False``.
        do_seq_class_task: whether or not to perform sequence level
    classifcation task. Default: ``False``.
        do_mult_choice_task: whether or not to perform multiple choice
    classifcation task. Default: ``False``.
        do_tok_class_task: whether or not to perform token level
    classifcation task. Default: ``False``.
        do_qa_task: whether or not to perform Question Answering
    prediction task. Default: ``False``.


        seq_class_num_labels: number of labels for the sequence level
    classifcation task. Default: 2.
        task_num_choices: number of choices for the multiple choice
    classifcation task. Default: 2.
        tok_class_num_labels: number of labels for the token level
    classifcation task. Default: 2.
        task_dropout: dropout rate to be applied on various tasks. Default: 0.1.



    Inputs:
        ``seq``: the input sequence of dimension :math:`(N, L)`
        `token_type_ids` (only if num_segments > 0): types of the tokens
    of dimension `(N, L)` with values in range [0, num_segments[. E.g., for
    num_segments = 2, Type 0 corresponds to a `sentence A` and type 1
    corresponds to a `sentence B` token (see BERT paper for more details).
        ``pos_tokens``: the position tokens over the input sequence of
    dimension :math:`(N, L)`
        ``attn_mask`` (only if use_attn_mask is True): the attn_mask of dimension
    :math:`(N, 1, L, L)`.
        ``pad_mask`` (only if use_pad_mask is True): the pad_mask of dimension
    :math:`(N, L, 1)`.

    Outputs:
        ``result``: the result of the operation of dimension :math:`(N, L, d_out)`
        ``pooled``(only if use_pooler is True): the result of the pooler
    operation of dimension :math:`(N, d_out)`
        ``lm_seq``(only if use_masked_lm is True): the result of the masked LM task
    of dimension :math:`(N, L, vocab_size)`
        ``next_sp_tgt``(only if use_next_sp is True): the result of the next sentence
    prediction task of dimension :math:`(N, 2)`
        ``seq_out``(only if do_seq_class_task is True): the result of the sentence
    classification task of dimension :math:`(N, seq_class_num_labels)`
        ``mult_out``(only if do_mult_choice_task is True): the result of the multiple
    choice task of dimension :math:`(N//task_num_choices, task_num_choices)`
        ``tok_out``(only if do_tok_class_task is True): the result of the token
    classification task of dimension :math:`(N, L, tok_class_num_labels)`
        ``qa_out_start``(only if do_qa_task is True): the result of the QA prediction
    task of dimension :math:`(N, L)`
        ``qa_out_end``(only if do_qa_task is True): the result of the QA prediction
    task of dimension :math:`(N, L)`
    """

    def __init__(
            self, vocab_size=30000, n_layers=12, d_model=768, d_inner=768 * 4,
            n_head=12, d_k=64, d_v=64, d_out=768, max_len=512, num_segments=2,
            embedding_dropout=0.1, attention_dropout=0.1, residual_dropout=0.1,
            embedding_layer_norm=False, layer_norm_epsilon=1e-5, neg_inf=-1e9,
            trainable_pos_embedding=True, use_one_embedding_dropout=False,
            use_attn_mask=True, use_pad_mask=False, use_gelu=True,
            accurate_gelu=False,  use_pooler=False, use_masked_lm=False,
            use_next_sp=False, do_seq_class_task=False, do_mult_choice_task=False,
            do_tok_class_task=False, do_qa_task=False,
            seq_class_num_labels=2, task_num_choices=2, tok_class_num_labels=2,
            task_dropout=0.1, **kwargs):

        super(TransformerEncoder, self).__init__()
        if d_out is None:
            d_out = d_model
        if not use_pooler:
            use_next_sp = False
        if do_seq_class_task or do_mult_choice_task:
            assert use_pooler
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.max_len = max_len
        self.num_segments = num_segments
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_layer_norm = embedding_layer_norm
        self.layer_norm_epsilon = layer_norm_epsilon

        self.neg_inf = neg_inf
        self.trainable_pos_embedding = trainable_pos_embedding
        self.use_one_embedding_dropout = use_one_embedding_dropout
        self.use_attn_mask = use_attn_mask
        self.use_pad_mask = use_pad_mask
        self.use_gelu = use_gelu
        self.accurate_gelu = accurate_gelu
        self.use_pooler = use_pooler
        self.use_masked_lm = use_masked_lm
        self.use_next_sp = use_next_sp

        self.do_seq_class_task = do_seq_class_task
        self.do_mult_choice_task = do_mult_choice_task
        self.do_tok_class_task = do_tok_class_task
        self.do_qa_task = do_qa_task

        self.seq_class_num_labels = seq_class_num_labels
        self.task_num_choices = task_num_choices
        self.tok_class_num_labels = tok_class_num_labels
        self.task_dropout = task_dropout

        self.embed_layer = Embedding(
            output_dim=d_model, dropout=embedding_dropout, vocab_size=vocab_size,
            max_len=max_len, trainable_pos_embedding=trainable_pos_embedding,
            num_segments=num_segments,
            use_one_dropout=use_one_embedding_dropout,
            use_embedding_layer_norm=embedding_layer_norm,
            layer_norm_epsilon=layer_norm_epsilon
        )
        self.enc_layers = []
        for i in range(n_layers):
            self.enc_layers.append(
                EncoderLayer(
                    d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k,
                    d_v=d_v, d_out=d_out, residual_dropout=residual_dropout,
                    attention_dropout=attention_dropout,
                    use_pad_mask=use_pad_mask, use_attn_mask=use_attn_mask,
                    neg_inf=neg_inf, ln_epsilon=layer_norm_epsilon,
                    use_gelu=use_gelu, accurate_gelu=accurate_gelu,
                    name='enc_layer_{}'.format(i)
                )
            )
        if self.use_pooler:
            self.pooler = Pooler(self.d_out)

        if self.use_masked_lm or self.use_next_sp:
            emb_p = None
            if self.use_masked_lm:
                # self.embed_layer.token_emb.build((None, None))
                # emb_p = K.transpose(self.embed_layer.token_emb.embeddings)
                emb_p = self.embed_layer.token_emb
            self.task_cls = MaskedLM_NextSentenceHead(
                self.d_out, self.vocab_size,
                embedding_weights=emb_p,
                layer_norm_epsilon=layer_norm_epsilon,
                use_gelu=use_gelu, accurate_gelu=accurate_gelu,
                use_masked_lm=self.use_masked_lm, use_next_sp=self.use_next_sp
            )

        if do_seq_class_task:
            self.seq_class_task = SequenceClassificationTask(
                self.d_out, seq_class_num_labels, task_dropout
            )

        if do_mult_choice_task:
            self.mult_choice_task = MultipleChoiceTask(
                self.d_out, task_num_choices, task_dropout
            )

        if do_tok_class_task:
            self.tok_class_task = TokenClassificationTask(
                self.d_out, tok_class_num_labels, task_dropout
            )

        if do_qa_task:
            self.qa_task = QuestionAnsweringTask(
                self.d_out, task_dropout
            )

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[0]).as_list()
        shape.append(self.d_out)

        ret = []
        if not self.use_pooler:
            if not self.use_masked_lm:
                ret = [tf.TensorShape(shape)]
            else:
                shape3 = [shape[0], shape[1], self.vocab_size]
                ret = [tf.TensorShape(shape), tf.TensorShape(shape3)]
        else:
            shape2 = [shape[0], shape[2]]
            if not self.use_masked_lm and not self.use_next_sp:
                ret = [tf.TensorShape(shape), tf.TensorShape(shape2)]
            elif not self.use_masked_lm:
                shape3 = [shape[0], 2]
                ret = [
                    tf.TensorShape(shape), tf.TensorShape(shape2),
                    tf.TensorShape(shape3)
                ]
            elif not self.use_next_sp:
                shape3 = [shape[0], shape[1], self.vocab_size]
                ret = [
                    tf.TensorShape(shape), tf.TensorShape(shape2),
                    tf.TensorShape(shape3)
                ]
            else:
                shape3 = [shape[0], shape[1], self.vocab_size]
                shape4 = [shape[0], 2]
                ret = [
                    tf.TensorShape(shape), tf.TensorShape(shape2),
                    tf.TensorShape(shape3), tf.TensorShape(shape4)
                ]

        if self.do_seq_class_task:
            shape_seq = [shape[0], self.seq_class_num_labels]
            ret.append(tf.TensorShape(shape_seq))

        if self.do_mult_choice_task:
            shape_mult = [-1, self.task_num_choices]
            ret.append(tf.TensorShape(shape_mult))

        if self.do_tok_class_task:
            shape_tok = [shape[0], shape[1], self.tok_class_num_labels]
            ret.append(tf.TensorShape(shape_tok))

        if self.do_qa_task:
            shape_qa = [shape[0], shape[1]]
            ret.append(tf.TensorShape(shape_qa))

        if len(ret) == 1:
            ret = ret[0]

        return ret

    def call(self, inputs):

        if self.num_segments is None or (self.num_segments == 0):
            segment_ids = None
            if not self.use_attn_mask and not self.use_pad_mask:
                tokens, pos_ids = inputs
                attn_mask = None
                pad_mask = None
            elif not self.use_pad_mask:
                tokens, pos_ids, attn_mask = inputs
                pad_mask = None
            elif not self.use_attn_mask:
                tokens, pos_ids, pad_mask = inputs
                attn_mask = None
            else:
                tokens, pos_ids, attn_mask, pad_mask = inputs
        else:
            if not self.use_attn_mask and not self.use_pad_mask:
                tokens, segment_ids, pos_ids = inputs
                attn_mask = None
                pad_mask = None
            elif not self.use_pad_mask:
                tokens, segment_ids, pos_ids, attn_mask = inputs
                pad_mask = None
            elif not self.use_attn_mask:
                tokens, segment_ids, pos_ids, pad_mask = inputs
                attn_mask = None
            else:
                tokens, segment_ids, pos_ids, attn_mask, pad_mask = inputs

        args = [tokens, pos_ids] if segment_ids is None else [
            tokens, segment_ids, pos_ids
        ]
        if len(args) == 1:
            args = args[0]
        x = self.embed_layer(args)
        for i in range(len(self.enc_layers)):
            args = [x]
            if self.use_attn_mask:
                args.append(attn_mask)
            if self.use_pad_mask:
                args.append(pad_mask)
            if len(args) == 1:
                args = args[0]
            x, _ = self.enc_layers[i](args)

        ret = []
        if not self.use_pooler:
            if not self.use_masked_lm:
                ret = [x]
            else:
                lm_seq = self.task_cls(x)
                ret = [x, lm_seq]
        else:
            pooled = self.pooler(x)
            if not self.use_masked_lm and not self.use_next_sp:
                ret = [x, pooled]
            elif not self.use_masked_lm:
                next_sp_tgt = self.task_cls(pooled)
                ret = [x, pooled, next_sp_tgt]
            elif not self.use_next_sp:
                lm_seq = self.task_cls(x)
                ret = [x, pooled, lm_seq]
            else:
                lm_seq, next_sp_tgt = self.task_cls([x, pooled])
                ret = [x, pooled, lm_seq, next_sp_tgt]

        if self.do_seq_class_task:
            seq_out = self.seq_class_task(pooled)
            ret.append(seq_out)

        if self.do_mult_choice_task:
            mult_out = self.mult_choice_task(pooled)
            ret.append(mult_out)

        if self.do_tok_class_task:
            tok_out = self.tok_class_task(x)
            ret.append(tok_out)

        if self.do_qa_task:
            qa_out = self.qa_task(x)
            ret.extend(qa_out)

        if len(ret) == 1:
            ret = ret[0]

        return ret

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'n_head': self.n_head,
            'd_k': self.d_k,
            'd_v': self.d_v,
            'd_out': self.d_out,
            'max_len': self.max_len,
            'num_segments': self.num_segments,
            'embedding_dropout': self.embedding_dropout,
            'attention_dropout': self.attention_dropout,
            'residual_dropout': self.residual_dropout,
            'embedding_layer_norm': self.embedding_layer_norm,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'neg_inf': self.neg_inf,
            'trainable_pos_embedding': self.trainable_pos_embedding,
            'use_one_embedding_dropout': self.use_one_embedding_dropout,
            'use_attn_mask': self.use_attn_mask,
            'use_pad_mask': self.use_pad_mask,
            'use_gelu': self.use_gelu,
            'accurate_gelu': self.accurate_gelu,
            'use_pooler': self.use_pooler,
            'use_masked_lm': self.use_masked_lm,
            'use_next_sp': self.use_next_sp,
            'do_seq_class_task': self.do_seq_class_task,
            'do_mult_choice_task': self.do_mult_choice_task,
            'do_tok_class_task': self.do_tok_class_task,
            'do_qa_task': self.do_qa_task,
            'seq_class_num_labels': self.seq_class_num_labels,
            'task_num_choices': self.task_num_choices,
            'tok_class_num_labels': self.tok_class_num_labels,
            'task_dropout': self.task_dropout,

        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == '__main__':

    tf.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    params = {
        'vocab_size': 20,
        'n_layers': 3,
        'd_model': 15,
        'd_inner': 50,
        'n_head': 5,
        'd_k': 15,
        'd_v': 15,
        'd_out': 15,
        'max_len': 10,
        'num_segments': 0,
        'embedding_dropout': 0.1,
        'attention_dropout': 0.1,
        'residual_dropout': 0.1,
        'embedding_layer_norm': False,
        'layer_norm_epsilon': 1e-5,
        'neg_inf': -1e9,
        'trainable_pos_embedding': True,
        'use_one_embedding_dropout': True,
        'use_attn_mask': False,
        'use_pad_mask': False,
        'use_gelu': False,
        'accurate_gelu': False,
        'use_pooler': True,
        'use_masked_lm': True,
        'use_next_sp': True,

        'do_seq_class_task': True,
        'do_mult_choice_task': True,
        'do_tok_class_task': True,
        'do_qa_task': True,
        'seq_class_num_labels': 3,
        'task_num_choices': 2,  # bath_size must be a multiple of this params
        'tok_class_num_labels': 5,
        'task_dropout': 0.1,
    }

    model = TransformerEncoder(**params)

    tokens = tf.Variable([[1, 5, 4, 3, 2, 10, 15], [1, 5, 8, 3, 9, 10, 15]])
    pos_ids = tf.Variable([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])
    outputs = model([tokens, pos_ids])
    if isinstance(outputs, (list, tuple)):
        out = ['Output_{}: {}'.format(i, v.shape)
               for i, v in enumerate(outputs)]
        print(' '.join(out))
    else:
        print('Output: ', outputs.shape)

    tokens = tf.Variable([[1, 5, 4, 3, 2, 10, 15, 18],
                          [1, 5, 16, 3, 2, 14, 15, 18]])
    pos_ids = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]])
    outputs = model([tokens, pos_ids])

    if isinstance(outputs, (list, tuple)):
        out = ['Output2_{}: {}'.format(i, v.shape)
               for i, v in enumerate(outputs)]
        print(' '.join(out))
    else:
        print('Output2: ', outputs.shape)

    print('done!')
