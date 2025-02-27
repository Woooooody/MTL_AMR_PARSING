# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from thumt.layers.nn import linear


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def split_heads(inputs, num_heads, name=None):
    """ Split heads
    :param inputs: A tensor with shape [batch, ..., channels]
    :param num_heads: An integer
    :param name: An optional string
    :returns: A tensor with shape [batch, heads, ..., channels / heads]
    """

    with tf.name_scope(name, default_name="split_heads", values=[inputs]):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims
        ndims = x.shape.ndims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        perm = [0, ndims - 1] + [i for i in range(1, ndims - 1)] + [ndims]
        return tf.transpose(ret, perm)


def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: one of "causal", "masking", "proximal" or "distance"
    :param inf: A floating value
    :param name: optional string
    :returns: A 4D tensor with shape [batch, heads, queries, memories]
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)


def should_generate_summaries():
    """Is this an appropriate context to generate summaries.
    :returns: a boolean
    """
    if "while/" in tf.contrib.framework.get_name_scope():
        return False
    if tf.get_variable_scope().reuse:
        return False
    return True


def attention_image_summary(weights, rgb=True):
    """Compute attention image summary.
    :param weights: a Tensor with shape [batch, heads, queries, memories]
    :param rgb: use RGB color to represent a head
    """
    shape = tf.shape(weights)
    batch_size = shape[0]
    num_heads = shape[1]
    num_queries = shape[2]
    num_memories = shape[3]

    if rgb:
        # [batch, queries, memories, heads]
        image = tf.transpose(weights, [0, 2, 3, 1])
        # for high-dynamic-range
        image = tf.pow(image, 0.2)
        # Each head will correspond to one of RGB
        image = tf.pad(image, [[0, 0], [0, 0], [0, 0],
                               [0, tf.mod(-num_heads, 3)]])
        shape = tf.shape(image)
        # [batch, queries, memories, 3, heads]
        image = tf.reshape(image, [batch_size, num_queries, num_memories,
                                   3, shape[-1] // 3])
        image = tf.reduce_max(image, 4)
    else:
        image = tf.reshape(weights, [-1, num_queries, num_memories, 1])

    # [batch, height, width, channel]
    tf.summary.image("attention", image, max_outputs=1)


def attention(query, memories, bias, hidden_size, cache=None, reuse=None,
              dtype=None, scope=None):
    """ Standard attention layer

    :param query: A tensor with shape [batch, key_size]
    :param memories: A tensor with shape [batch, memory_size, key_size]
    :param bias: A tensor with shape [batch, memory_size]
    :param hidden_size: An integer
    :param cache: A dictionary of precomputed value
    :param reuse: A boolean value, whether to reuse the scope
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer
    :return: A tensor with shape [batch, value_size] and
        a Tensor with shape [batch, memory_size]
    """

    with tf.variable_scope(scope or "attention", reuse=reuse,
                           values=[query, memories, bias], dtype=dtype):
        mem_shape = tf.shape(memories)
        key_size = memories.get_shape().as_list()[-1]

        if cache is None:
            k = tf.reshape(memories, [-1, key_size])
            k = linear(k, hidden_size, False, False, scope="k_transform")

            if query is None:
                return {"key": k}
        else:
            k = cache["key"]

        q = linear(query, hidden_size, False, False, scope="q_transform")
        k = tf.reshape(k, [mem_shape[0], mem_shape[1], hidden_size])

        hidden = tf.tanh(q[:, None, :] + k)
        hidden = tf.reshape(hidden, [-1, hidden_size])

        # Shape: [batch, mem_size, 1]
        logits = linear(hidden, 1, False, False, scope="logits")
        logits = tf.reshape(logits, [-1, mem_shape[1]])

        if bias is not None:
            logits = logits + bias

        alpha = tf.nn.softmax(logits)

        outputs = {
            "value": tf.reduce_sum(alpha[:, :, None] * memories, axis=1),
            "weight": alpha
        }

    return outputs


def additive_attention(queries, keys, values, bias, hidden_size, concat=False,
                       keep_prob=None, dtype=None, scope=None):
    """ Additive attention mechanism. This layer is implemented using a
        one layer feed forward neural network

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param hidden_size: An integer
    :param concat: A boolean value. If ``concat'' is set to True, then
        the computation of attention mechanism is following $tanh(W[q, k])$.
        When ``concat'' is set to False, the computation is following
        $tanh(Wq + Vk)$
    :param keep_prob: a scalar in [0, 1]
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string, the scope of this layer

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, length_q]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.variable_scope(scope, default_name="additive_attention",
                           values=[queries, keys, values, bias], dtype=dtype):
        length_q = tf.shape(queries)[2]
        length_kv = tf.shape(keys)[2]
        q = tf.tile(tf.expand_dims(queries, 3), [1, 1, 1, length_kv, 1])
        k = tf.tile(tf.expand_dims(keys, 2), [1, 1, length_q, 1, 1])

        if concat:
            combined = tf.tanh(linear(tf.concat([q, k], axis=-1), hidden_size,
                                      True, True, name="qk_transform"))
        else:
            q = linear(queries, hidden_size, True, True, name="q_transform")
            k = linear(keys, hidden_size, True, True, name="key_transform")
            combined = tf.tanh(q + k)

        # shape: [batch, heads, length_q, length_kv]
        logits = tf.squeeze(linear(combined, 1, True, True, name="logits"),
                            axis=-1)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob or keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multiplicative_attention(queries, keys, values, bias, keep_prob=None,
                             name=None):
    """ Multiplicative attention mechanism. This layer is implemented using
        dot-product operation.

    :param queries: A tensor with shape [batch, heads, length_q, depth_k]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param keep_prob: a scalar in (0, 1]
    :param name: the name of this operation

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, heads, length_q, depth_v]
    """

    with tf.name_scope(name, default_name="multiplicative_attention",
                       values=[queries, keys, values, bias]):
        # shape: [batch, heads, length_q, length_kv]
        logits = tf.matmul(queries, keys, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        outputs = tf.matmul(weights, values)

        return {"weights": weights, "outputs": outputs}


def multihead_attention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, keep_prob=None, output=True,
                        state=None, summary=True, dtype=None, scope=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param state: An optional dictionary used for incremental decoding
    :param summary: Use image summary
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[queries, memories], dtype=dtype, reuse=tf.AUTO_REUSE):
        next_state = {}

        if memories is None:
            # self attention
            size = key_size * 2 + value_size    # encoder中,q k v是一个东西

            combined = linear(queries, size, True, True, scope=scope+"_qkv_transform")     # (?, ?, 1536)

            q, k, v = tf.split(combined, [key_size, key_size, value_size],      # (?, ?, 512)
                               axis=-1)

            if state is not None:
                k = tf.concat([state["key"], k], axis=1)
                v = tf.concat([state["value"], v], axis=1)
                next_state["key"] = k
                next_state["value"] = v
        else:
            q = linear(queries, key_size, True, True, scope=scope+"_q_transform")
            combined = linear(memories, key_size + value_size, True,
                              scope=scope+"_kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)

        # split heads
        q = split_heads(q, num_heads)   # (?, 8, ?, 64)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)


        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        results = multiplicative_attention(q, k, v, bias, keep_prob)    # (?, 8, ?, 64)

        # combine heads
        weights = results["weights"]
        x = combine_heads(results["outputs"])   # (?, ?, 512)

        if output:
            outputs = linear(x, output_size, True, True,
                             scope=scope+"_output_transform")
        else:
            outputs = x

        if should_generate_summaries() and summary:
            attention_image_summary(weights)

        outputs = {"weights": weights, "outputs": outputs}

        if state is not None:
            outputs["state"] = next_state

        return outputs
