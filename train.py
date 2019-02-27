import time
import pprint
import tensorflow as tf
import argparse
import os
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

tf.app.flags.DEFINE_boolean('with_model', False, 'Continue from previously saved model')

FLAGS = flags.FLAGS

def main(_):
    pprint.pprint(FLAGS.flag_values_dict())

    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    else:
        if FLAGS.with_model:
            old_model_checkpoint_path = open('saved_model/checkpoint', 'r')
            old_model_checkpoint_path = "".join(["saved_model/",old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])

    print("Initializing dataset ...")
    db = DB(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs,
        shuffle_buffer=131072 , map_parallel=4, step='train')

    dataset = db.get_dataset()
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types, dataset.output_shapes)
    next_element = iterator.get_next()
    train_iterator = dataset.make_one_shot_iterator()

    summary_list = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Initializing model ...")
        model = Model(FLAGS)
        train_handle = sess.run(train_iterator.string_handle())

        loss_summary = tf.summary.scalar('loss', model.loss)
        summary = tf.summary.merge([loss_summary])
        writer = tf.summary.FileWriter('./saved_model', sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        if 'old_model_checkpoint_path' in globals():
            print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
            saver.restore(sess, old_model_checkpoint_path )


        print("\nIteration starts.")
        start = time.time()
        for epoch in range(FLAGS.num_epochs):
            for ind_batch in range(3803957/FLAGS.batch_size):
                batch_x, batch_x_len, batch_decoder_input, batch_decoder_output, batch_decoder_len = \
                        sess.run(next_element, feed_dict={handle: train_handle})
                train_feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: batch_x,
                    model.X_len: batch_x_len,
                    model.decoder_input: batch_decoder_input,
                    model.decoder_len: batch_decoder_len,
                    model.decoder_target: batch_decoder_output
                }

                _, step, loss, summary_val = sess.run([model.update, model.global_step, model.loss, summary], feed_dict=train_feed_dict)
                
                if step % 100 == 0:
                    writer.add_summary(summary_val, model.global_step.eval())
                    seconds = time.time()-start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print("\tepoch {:02d} step {:06d}:\tloss = {:.3e}\ttime = {:02d}h{:02d}m{:.2f}s".format(epoch, step, loss, int(h),int(m),s))

            saver.save(sess, "./saved_model/model.ckpt", global_step=step)
        writer.close()

if __name__ == '__main__':
    tf.app.run()