import tensorflow as tf

f1 = "../data/corpus.tc.en"
f2 = "../data/corpus.tc.de"

data1 = tf.data.TextLineDataset(f1)
data2 = tf.data.TextLineDataset(f2)

dataset = tf.data.Dataset.zip((data1, data2))

dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=1
        )

dataset = dataset.map(
            lambda src, tgt: (
                tf.concat([src, [tf.constant("<eos>")]], axis=0),
                tf.concat([tgt, [tf.constant("<eos>")]], axis=0)
            ),
            num_parallel_calls=1
        )

dataset = dataset.map(
            lambda src, tgt: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=1
        )

iterator = dataset.make_one_shot_iterator()
features = iterator.get_next()

sess = tf.Session()
s = sess.run(features)
print(s)