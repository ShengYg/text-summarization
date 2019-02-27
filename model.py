import tensorflow as tf
import os
import cPickle
from utils import get_init_embedding


class Model(object):
    def __init__(self, args, article_max_len=50, summary_max_len=15, forward_only=False):
        if os.path.exists("./data/sumdata/word_dict.pkl"):
            with open("./data/sumdata/word_dict.pkl", "rb") as f:
                word_dict = cPickle.load(f)
        self.reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
        self.vocabulary_size = len(self.reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        with tf.variable_scope("decoder/projection"):
            # tf.glorot_normal_initializer??
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            if not forward_only and args.glove:
                print("Using glove embedding...")
                init_embeddings = tf.constant(get_init_embedding(), dtype=tf.float32)
            else:
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1, 0, 2])

        with tf.name_scope("encoder"):

            lstm  = tf.contrib.cudnn_rnn.CudnnLSTM(2, self.num_hidden, 
                direction="bidirectional", kernel_initializer=tf.orthogonal_initializer())
            self.encoder_output, state = lstm(self.encoder_emb_inp)
            state_h, state_c = state
            c_fw, c_bw = state_c[0, :], state_c[1, :]
            h_fw, h_bw = state_h[0, :], state_h[1, :]
            state_fw, state_bw = h_fw, h_bw
            self.encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                c=tf.concat((c_fw, c_bw), 1), 
                h=tf.concat((h_fw, h_bw), 1))

            # fw_cells = [tf.nn.rnn_cell.LSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            # bw_cells = [tf.nn.rnn_cell.LSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            # fw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell) for cell in fw_cells]
            # bw_cells = [tf.nn.rnn_cell.DropoutWrapper(cell) for cell in bw_cells]

            # encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #     fw_cells, bw_cells, self.encoder_emb_inp,
            #     sequence_length=self.X_len, time_major=True, dtype=tf.float32)
            # self.encoder_output = tf.concat(encoder_outputs, 2)
            # encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            # encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            # self.encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden * 2,
                initializer=tf.orthogonal_initializer())
            # decoder_cell  = tf.contrib.cudnn_rnn.CudnnLSTM(1, self.num_hidden * 2, 
            #     direction="unidirectional", kernel_initializer=tf.orthogonal_initializer())

            if not forward_only:
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2,
                                                                   output_attention=False)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat(
                    [self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2,
                                                                   output_attention=False)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
