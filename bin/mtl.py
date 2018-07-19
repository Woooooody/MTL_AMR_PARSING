#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six

import numpy as np
import tensorflow as tf
import thumt.data.cache as cache
import thumt.data.dataset as dataset
import thumt.data.record as record
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimize as optimize
import thumt.utils.parallel as parallel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--parsing_input", type=str, nargs=2,
                        help="Path of source and target corpus"
                        )
    parser.add_argument("--amr_input", type=str, nargs=2,
                        help="Path of source and target corpus"
                        )
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str, nargs=3,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")


    # model and configuration
    parser.add_argument("--model", type=str,
                        help="Name of the model", default="transformer")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    # print(parser.parse_args(args))

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        parsing_input=["../data/amr/syntax/source.train",    # source input file
                        "../data/amr/syntax/parsing.linear.train"],                          # parsing target file
        amr_input=["../data/amr/baselineamr/train_source.txt",  # source input file
                    "../data/amr/baselineamr/train_target.txt"],
        output="",
        record="",
        model="transformer",
        vocab=["../data/amr/syntax/vocab.mtl.source.txt",
               "../data/amr/syntax/vocab.parsing.linear.txt",
               "../data/amr/baselineamr/vocab.amr.target"],
        # Default training hyper parameters
        num_threads=6,
        batch_size=4096,
        max_length=512,  # 256
        length_multiplier=1,
        mantissa_bits=2,
        warmup_steps=16000,  # 4000
        train_steps=20000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        update_cycle=1,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        scale_l1=0.0,
        scale_l2=0.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.997,   # 0.999
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        # Validation
        eval_steps=2,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        validation=None,
        references=None,#["../data/amr/baselineamr/small.target.dev"],
        save_checkpoint_secs=0,
        save_checkpoint_steps=200,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().iterkeys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.parsing_input = args.parsing_input or params.parsing_input
    params.amr_input = args.amr_input or params.amr_input
    params.output = args.output or params.output
    params.record = args.record or params.record
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocab[0]),
        "parsing_target": vocabulary.load_vocabulary(params.vocab[1]),
        "amr_target": vocabulary.load_vocabulary(params.vocab[2])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["parsing_target"] = vocabulary.process_vocabulary(
        params.vocabulary["parsing_target"], params
    )
    params.vocabulary["amr_target"] = vocabulary.process_vocabulary(
        params.vocabulary["amr_target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "parsing_target": vocabulary.get_control_mapping(
            params.vocabulary["parsing_target"],
            control_symbols
        ),
        "amr_target": vocabulary.get_control_mapping(
            params.vocabulary["amr_target"],
            control_symbols
        )
    }

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def decode_target_ids(inputs, params, flag='target'):
    decoded = []
    vocab = params.vocabulary[flag]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)
    return decoded


def decode_source_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["source"]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)
    return decoded


def restore_variables(checkpoint):
    if not checkpoint:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return tf.group(*ops, name="restore_op")


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)  # a model class
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )
    # print(params.vocabulary)

    # Build Graph
    with tf.Graph().as_default():
        if not params.record:
            # Build input queue
            parsing_features = dataset.get_training_input(params.parsing_input, params, problem='parsing')
            amr_features = dataset.get_training_input(params.amr_input, params, problem='amr')
        else:
            parsing_features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )
            amr_features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )

        update_cycle = params.update_cycle
        parsing_features, parsing_init_op = cache.cache_features(parsing_features, update_cycle)
        amr_features, amr_init_op = cache.cache_features(amr_features, update_cycle)

        # Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        model = model_cls(params)
        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Multi-GPU setting
        # parsing_sharded_losses, amr_sharded_losses = parallel.parallel_model(
        #         model.get_training_func(initializer, regularizer),
        #         [parsing_features, amr_features],
        #         params.device_list
        # )
        # parsing_loss, amr_loss = model.get_training_func(initializer, regularizer)([parsing_features, amr_features])
        # with tf.variable_scope("shared_decode_variable") as scope:
        #     parsing_loss = model.get_training_func(initializer, regularizer, problem='parsing')(parsing_features)
        #     scope.reuse_variables()
        #     amr_loss = model.get_training_func(initializer, regularizer, problem='amr')(amr_features)
        #with tf.variable_scope("encoder_shared", reuse=True):
        print(params.layer_postprocess)
        with tf.variable_scope("encoder_shared", initializer=initializer,
                               regularizer=regularizer, reuse=tf.AUTO_REUSE):
            parsing_encoder_output = model.get_encoder_out(parsing_features,
                                                           "train",
                                                           params)
            amr_encoder_output = model.get_encoder_out(amr_features,
                                                       "train",
                                                       params)
        with tf.variable_scope("parsing_decoder", initializer=initializer,
                               regularizer=regularizer):
            parsing_loss = model.get_decoder_out(parsing_features,
                                                 parsing_encoder_output,
                                                 "train",
                                                 params,
                                                 problem="parsing")
        with tf.variable_scope("amr_decoder", initializer=initializer,
                               regularizer=regularizer):
            amr_loss = model.get_decoder_out(amr_features,
                                             amr_encoder_output,
                                             "train",
                                             params,
                                             problem="amr")


        parsing_loss = parsing_loss + tf.losses.get_regularization_loss()

        amr_loss = amr_loss + tf.losses.get_regularization_loss()

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        parsing_loss, parsing_ops = optimize.create_train_op(parsing_loss, opt, global_step, params, problem="parsing")
        amr_loss, amr_ops = optimize.create_train_op(amr_loss, opt, global_step, params, problem="amr")

        restore_op = restore_variables(args.checkpoint)

        # Validation
        if params.validation and params.references[0]:
            files = [params.validation] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            eval_input_fn = dataset.get_evaluation_input
        else:
            eval_input_fn = None

        # Add hooks
        save_vars = tf.trainable_variables() + [global_step]
        saver = tf.train.Saver(
            var_list=save_vars if params.only_save_trainable else None,
            max_to_keep=params.keep_checkpoint_max,
            sharded=False
        )
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

        multiplier = tf.convert_to_tensor([update_cycle, 1])

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(parsing_loss),
            tf.train.NanTensorHook(amr_loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "parsing_loss": parsing_loss,
                    "amr_loss": amr_loss,
                    "parsing_source": tf.shape(parsing_features["source"]) * multiplier,
                    "parsing_target": tf.shape(parsing_features["target"]) * multiplier,
                    "amr_source": tf.shape(amr_features["source"]) * multiplier,
                    "amr_target": tf.shape(amr_features["target"]) * multiplier
                },
                every_n_iter=1
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=params.output,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=saver
            )
        ]

        config = session_config(params)

        if eval_input_fn is not None:
            train_hooks.append(
                hooks.EvaluationHook(
                    lambda f: inference.create_inference_graph(
                        [model], f, params
                    ),
                    lambda: eval_input_fn(eval_inputs, params),
                    lambda x: decode_target_ids(x, params, flag='target'),
                    lambda x: decode_target_ids(x, params, flag='source'),
                    params.output,
                    config,
                    params.keep_top_checkpoint_max,
                    eval_secs=params.eval_secs,
                    eval_steps=params.eval_steps
                )
            )

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        def parsing_step_fn(step_context):
            # Bypass hook calls
            step_context.session.run([parsing_init_op, parsing_ops["zero_op"]])  # if params.cycle==1 do nothing
            for i in range(update_cycle - 1):
                step_context.session.run(parsing_ops["collect_op"])

            return step_context.run_with_hooks(parsing_ops["train_op"])

        def amr_step_fn(step_context):
            # Bypass hook calls
            step_context.session.run([amr_init_op, amr_ops["zero_op"]])
            for i in range(update_cycle - 1):
                step_context.session.run(amr_ops["collect_op"])

            return step_context.run_with_hooks(amr_ops["train_op"])

        def step_fn(step_context):
            # Bypass hook calls
            return step_context.run_with_hooks(parsing_ops["train_op"])

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)
            step = 0
            while not sess.should_stop():
                if step % 2 == 0:
                    sess.run_step_fn(parsing_step_fn)
                else:
                    sess.run_step_fn(amr_step_fn)
                step += 1
                # sess.run_step_fn(step_fn)


if __name__ == "__main__":
    main(parse_args())
