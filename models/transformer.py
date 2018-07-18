# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    """
    :param inputs: word embedding of input sequence
    :return: attention output
    """
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs  # (?, ?, 512)

        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer, reuse=tf.AUTO_REUSE):
                with tf.variable_scope("self_attention", reuse=tf.AUTO_REUSE):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),     # _layer_process: either use norm for input
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        scope=scope+"_multihead_attention"
                    )

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward", reuse=tf.AUTO_REUSE):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        scope=scope + "_fnn"
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name=scope+"_decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = scope+"_layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope(scope+"_self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        scope=scope
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope(scope+"_encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        scope=scope
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope(scope+"_feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        scope=scope
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs  # (?, ?, 512)


def encoding_graph(features, mode, params):
    parsing_features = features[0]
    amr_features = features[1]
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    parsing_src_seq = parsing_features["source"]
    parsing_src_len = parsing_features["source_length"]
    parsing_src_mask = tf.sequence_mask(parsing_src_len,
                                        maxlen=tf.shape(parsing_features["source"])[1],
                                        dtype=tf.float32)

    amr_src_seq = amr_features["source"]
    amr_src_len = amr_features["source_length"]
    amr_src_mask = tf.sequence_mask(amr_src_len,
                                    maxlen=tf.shape(amr_features["source"])[1],
                                    dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    parsing_inputs = tf.gather(src_embedding, parsing_src_seq)  # lookup table the input embedding
    amr_inputs = tf.gather(src_embedding, amr_src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        parsing_inputs = parsing_inputs * (hidden_size ** 0.5)
        amr_inputs = amr_inputs * (hidden_size ** 0.5)

    parsing_inputs = parsing_inputs * tf.expand_dims(parsing_src_mask, -1)
    amr_inputs = amr_inputs * tf.expand_dims(amr_src_mask, -1)

    parsing_encoder_input = tf.nn.bias_add(parsing_inputs, bias)
    parsing_encoder_input = layers.attention.add_timing_signal(parsing_encoder_input)   # ??
    parsing_enc_attn_bias = layers.attention.attention_bias(parsing_src_mask, "masking")

    amr_encoder_input = tf.nn.bias_add(amr_inputs, bias)
    amr_encoder_input = layers.attention.add_timing_signal(amr_encoder_input)  # ??
    amr_enc_attn_bias = layers.attention.attention_bias(amr_src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        parsing_encoder_input = tf.nn.dropout(parsing_encoder_input, keep_prob)
        amr_encoder_input = tf.nn.dropout(amr_encoder_input, keep_prob)
    parsing_encoder_output = transformer_encoder(parsing_encoder_input, parsing_enc_attn_bias, params)
    amr_encoder_output = transformer_encoder(amr_encoder_input, amr_enc_attn_bias, params)

    return parsing_encoder_output, amr_encoder_output

def encoding_graph2(features, mode, params):

    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)  # lookup table the input embedding

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)   # ??
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, scope='shared')

    return encoder_output

def decoding_graph(features, state, mode, params, problem='parsing'):
    with tf.variable_scope(problem):
        if mode != "train":
            params.residual_dropout = 0.0
            params.attention_dropout = 0.0
            params.relu_dropout = 0.0
            params.label_smoothing = 0.0

        tgt_seq = features["target"]
        src_len = features["source_length"]
        tgt_len = features["target_length"]
        src_mask = tf.sequence_mask(src_len,
                                    maxlen=tf.shape(features["source"])[1],
                                    dtype=tf.float32)
        tgt_mask = tf.sequence_mask(tgt_len,
                                    maxlen=tf.shape(features["target"])[1],
                                    dtype=tf.float32)

        hidden_size = params.hidden_size
        if problem == 'parsing':
            tvocab = params.vocabulary["parsing_target"]
        elif problem == 'amr':
            tvocab = params.vocabulary["amr_target"]
        else:
            print("error ! problem must in parsing or amr !")
        tgt_vocab_size = len(tvocab)
        initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

        if params.shared_source_target_embedding:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                tgt_embedding = tf.get_variable(problem+"_weights",
                                                [tgt_vocab_size, hidden_size],
                                                initializer=initializer)
        else:
            tgt_embedding = tf.get_variable(problem+"_target_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)

        if params.shared_embedding_and_softmax_weights:
            weights = tgt_embedding
        else:
            weights = tf.get_variable(problem+"_softmax", [tgt_vocab_size, hidden_size],
                                      initializer=initializer)

        targets = tf.gather(tgt_embedding, tgt_seq)

        if params.multiply_embedding_mode == "sqrt_depth":
            targets = targets * (hidden_size ** 0.5)

        targets = targets * tf.expand_dims(tgt_mask, -1)

        enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
        dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                        "causal")
        # Shift left
        decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input = layers.attention.add_timing_signal(decoder_input)

        if params.residual_dropout:
            keep_prob = 1.0 - params.residual_dropout
            decoder_input = tf.nn.dropout(decoder_input, keep_prob)

        encoder_output = state["encoder"]

        if mode != "infer":
            decoder_output = transformer_decoder(decoder_input, encoder_output,
                                                 dec_attn_bias, enc_attn_bias,
                                                 params, scope=problem)
        else:
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]
            decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                                  dec_attn_bias, enc_attn_bias,
                                                  params, state=state["decoder"], scope=problem)

            decoder_output, decoder_state = decoder_outputs
            decoder_output = decoder_output[:, -1, :]
            logits = tf.matmul(decoder_output, weights, False, True)
            log_prob = tf.nn.log_softmax(logits)

            return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

        decoder_output = tf.reshape(decoder_output, [-1, hidden_size])  # (batch_size*seq_len, 512)

        logits = tf.matmul(decoder_output, weights, False, True)    # weights : (vocab_size, 512) ,logits : (?, vocab_size)
        labels = features["target"]


        # label smoothing
        ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            smoothing=params.label_smoothing,
            normalize=True
        )

        ce = tf.reshape(ce, tf.shape(tgt_seq))

        if mode == "eval":
            return -tf.reduce_sum(ce * tgt_mask, axis=1)

        loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    return loss


def model_graph(features, mode, params):
    parsing_encoder_output, amr_encoder_output = encoding_graph(features, mode, params)
    parsing_state = {"encoder": parsing_encoder_output}
    amr_state = {"encoder": parsing_encoder_output}
    parsing_output = decoding_graph(features[0], parsing_state, mode, params, problem='parsing')
    amr_output = decoding_graph(features[1], amr_state, mode, params, problem='amr')
    return parsing_output, amr_output

def model_graph2(features, mode, params, problem='parsing'):
    encoder_output = encoding_graph2(features, mode, params)
    state = {"encoder": encoder_output}
    output = decoding_graph(features, state, mode, params, problem=problem)
    return output

class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, problem=None):
        def training_fn(features, params=None, reuse=None, problem=problem):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                # parsing_loss, amr_loss = model_graph(features, "train", params)

                loss = model_graph2(features, "train", params, problem=problem)

                # encoder_output = encoding_graph2(features, "train", params)
                # amr_loss = model_graph(amr_features, "train", params, problem='amr')
                return loss

        return training_fn

    def get_encoder_out(self, features, mode, params, initializer=None, regularizer=None):
        with tf.variable_scope("encoder_shared", initializer=initializer,
                               regularizer=regularizer, reuse=tf.AUTO_REUSE):
            return encoding_graph2(features, mode, params)

    def get_decoder_out(self, features, encoder_out, mode, params, problem=None):
        state = {"encoder": encoder_out}
        return decoding_graph(features, state, mode, params, problem=problem)

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=1024,   # 512
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            attention_dropout=0.2,  # 0.0
            residual_dropout=0.2,   # 0.1
            relu_dropout=0.1,   # 0.0
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=0.05,     # 1.0
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.997,    # 0.98
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params
