from __future__ import print_function
from io import open
import tensorflow as tf
from model import Model
from utils import DB

flags = tf.app.flags

tf.app.flags.DEFINE_boolean('glove', False, 'Use glove as initial word embedding')
tf.app.flags.DEFINE_integer("embedding_size", 300, "Word embedding size")
tf.app.flags.DEFINE_integer("num_hidden", 150, "Network size")
tf.app.flags.DEFINE_integer("num_layers", 2, "Network depth")
tf.app.flags.DEFINE_integer("beam_width", 10, "Beam width for beam search decoder")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")

FLAGS = flags.FLAGS

def main(_):
    print("Initializing dataset ...")
    db = DB(batch_size=FLAGS.batch_size, num_epochs=1, 
        shuffle_buffer=131072, map_parallel=4, step='test')

    dataset = db.get_dataset()
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()
    train_iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        print("Loading saved model...")
        model = Model(FLAGS, forward_only=True)
        train_handle = sess.run(train_iterator.string_handle())
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './saved_model/model.ckpt-475488')

        print("Writing summaries to 'result.txt'...")
        try:
            while True:
                batch_x, batch_x_len = sess.run(next_element, feed_dict={handle: train_handle})
                write_batch_x = [[model.reversed_dict[y] for y in x if y != 0] for x in batch_x]

                valid_feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: batch_x,
                    model.X_len: batch_x_len,
                }

                prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
                write_batch_pred = [[model.reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

                with open("result.txt", "a") as f:
                    for (x, pred) in zip(write_batch_x, write_batch_pred):
                        print("Article: " + " ".join(x), file=f)

                        summary = []
                        for word in pred:
                            if word == "</s>":
                                break
                            if word not in summary:
                                summary.append(word)
                        print(" ".join(summary), file=f)
        except tf.errors.OutOfRangeError:
            pass

        print('Summaries are saved to "result.txt"...')

if __name__ == '__main__':
    tf.app.run()