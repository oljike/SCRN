# coding: utf-8


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import argparse
import copy
import collections
import math


class PTBSmallConfig(object):
    while True:
        # Global hyperparameters
        batch_size = np.random.choice([16, 32])
        max_grad_norm = 5
        lr_decay = round(np.random.uniform(0.5, 0.9), 2)
        learning_rate = round(np.random.uniform(0.1, 0.99), 2)
        init_scale = round(np.random.uniform(0.19, 0.5), 2)
        num_epochs = 90
        word_vocab_size = 10000  # to be determined later
        alpha = round(np.random.uniform(0.85, 0.99), 2)
        # RNN hyperparameters
        num_steps = np.random.randint(25, 80)
        hidden_size = 40
        context_size = 10

        if hidden_size > 0:
            break


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='PTBSmall',
                        help='config. Possible options: PTBSmall')
    parser.add_argument('--is_train', default='1',
                        help='mode. 1 = training, 0 = evaluation')
    parser.add_argument('--data_dir', default='data/ptb',
                        help='data directory. Should have train.txt/valid.txt' \
                             '/test.txt with input data')
    parser.add_argument('--save_dir', default='saves',
                        help='saves directory')
    parser.add_argument('--prefix', default='SCRN',
                        help='prefix for filenames when saving data and model')
    parser.add_argument('--eos', default='<eos>',
                        help='EOS marker')
    parser.add_argument('--verbose', default='1',
                        help='print intermediate results. 1 = yes, 0 = no')
    parser.add_argument('--gpu', default='0',
                        help='GPU ID to use.')
    return parser.parse_args()


def read_data(args, config):
    '''read data sets, construct all needed structures and update the config'''
    if args.is_train == '1':
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(os.path.join(
                args.save_dir, args.prefix + '-data.pkl'), 'wb') as data_file:
            word_data = open(os.path.join(args.data_dir, 'train.txt'), 'r').read() \
                .replace('\n', args.eos).split()
            counter = collections.Counter(word_data)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*count_pairs))

            pickle.dump(
                (word_data, words), data_file)

    else:
        with open(os.path.join(
                args.save_dir, args.prefix + '-data.pkl'), 'rb') as data_file:
            word_data, words = pickle.load(data_file)

    word_data_size, word_vocab_size = len(word_data), len(words)
    print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
    config.word_vocab_size = word_vocab_size

    word_to_ix = {word: i for i, word in enumerate(words)}
    ix_to_word = {i: word for i, word in enumerate(words)}

    def get_word_raw_data(input_file):
        data = open(input_file, 'r').read().replace('\n', args.eos).split()
        return [word_to_ix[w] for w in data]

    train_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'train.txt'))
    valid_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'valid.txt'))
    test_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'test.txt'))

    return train_raw_data, valid_raw_data, test_raw_data


class my_batch_producer(object):
    '''Slice the raw data into batches'''

    def __init__(self, raw_data, batch_size, num_steps):
        self.raw_data = raw_data
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.batch_len = len(self.raw_data) // self.batch_size
        self.data = np.reshape(self.raw_data[0: self.batch_size * self.batch_len],
                               (self.batch_size, self.batch_len))

        self.epoch_size = (self.batch_len - 1) // self.num_steps
        self.i = 0

    def __next__(self):
        if self.i < self.epoch_size:
            # batch_x and batch_y are of shape [batch_size, num_steps]
            batch_x = self.data[::,
                      self.i * self.num_steps: (self.i + 1) * self.num_steps:]
            batch_y = self.data[::,
                      self.i * self.num_steps + 1: (self.i + 1) * self.num_steps + 1:]
            self.i += 1
            return (batch_x, batch_y)
        else:
            raise StopIteration()

    def __iter__(self):
        return self


# new activation
def gw(u):
    pi = math.pi
    return tf.cast(tf.greater_equal(u, -pi / 2), tf.float32) * tf.cast(tf.less(u, pi / 2), tf.float32) * 0.5 * (
                tf.cos(u + (3 * pi / 2)) + 1) + (1 - tf.cast(tf.less(u, pi / 2), tf.float32))


class SCRNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, hidden_size, context_size, alpha, activation=tf.nn.sigmoid):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._context_size = context_size
        self._alpha = alpha
        self._activation = activation

    @property
    def state_size(self):
        return (self._context_size, self._hidden_size)

    @property
    def output_size(self):
        return (self._context_size, self._hidden_size)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "scrn_cell"):
            # inputs = tf.cast(inputs, tf.int32)
            s, h = state

            self.B = tf.get_variable('B', [self._input_size, self._context_size],
                                     dtype=tf.float32)
            self.P = tf.get_variable('P', [self._context_size, self._hidden_size],
                                     dtype=tf.float32)
            self.A = tf.get_variable('A', [self._input_size, self._hidden_size],
                                     dtype=tf.float32)
            self.R = tf.get_variable('R', [self._hidden_size, self._hidden_size],
                                     dtype=tf.float32)
            self.b = tf.get_variable('b', [self._hidden_size], dtype=tf.float32)

            new_s = tf.multiply(1 - self._alpha, tf.nn.embedding_lookup(self.B, inputs)) \
                    + tf.multiply(self._alpha, s)
            new_h = gw(
                tf.matmul(new_s, self.P) + tf.nn.embedding_lookup(self.A, inputs) + tf.matmul(h, self.R) + self.b)

        return (new_s, new_h), (new_s, new_h)


class Model:
    '''Word-level language model'''

    def __init__(self, config, need_reuse=False):
        # get hyperparameters
        batch_size = config.batch_size
        num_steps = config.num_steps
        self.init_scale = init_scale = config.init_scale
        hidden_size = config.hidden_size
        context_size = config.context_size
        word_vocab_size = config.word_vocab_size
        alpha = config.alpha

        # placeholders for training data and labels
        self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
        y_float = tf.cast(self.y, tf.float32)

        # ... and then process it with a stack of two LSTMs
        rnn_input = tf.unstack(self.x, axis=1)
        # basic SCRN cell
        self.cell = SCRNCell(word_vocab_size, hidden_size, context_size, alpha)

        self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
        with tf.variable_scope('lstm_rnn', reuse=need_reuse):
            outputs, self.state = tf.nn.static_rnn(
                self.cell,
                rnn_input,
                dtype=tf.float32,
                initial_state=self.init_state)

        if not need_reuse:
            self.state = outputs[4]

        # finally we predict the next word according to a softmax normalization
        with tf.variable_scope('softmax_params', reuse=need_reuse):
            U = tf.get_variable('U', [context_size, word_vocab_size],
                                dtype=tf.float32)
            V = tf.get_variable('V', [hidden_size, word_vocab_size],
                                dtype=tf.float32)
            b = tf.get_variable('b', [word_vocab_size],
                                dtype=tf.float32)

        outputs_s, outputs_h = zip(*outputs)
        output_s = tf.reshape(tf.concat(outputs_s, 1), [-1, context_size])
        output_h = tf.reshape(tf.concat(outputs_h, 1), [-1, hidden_size])
        logits = tf.matmul(output_s, U) + tf.matmul(output_h, V) + b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.y, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size


class Train(Model):
    '''for training we need to compute gradients'''

    def __init__(self, config):
        super(Train, self).__init__(config)

        self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=tf.train.get_or_create_global_step())

        self.new_lr = tf.placeholder(tf.float32, shape=[],
                                     name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)

    # this will update the learning rate
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def model_size():
    '''finds the total number of trainable variables a.k.a. model size'''
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


def run_epoch(sess, model, raw_data, config, is_train=False, lr=None):
    start_time = time.time()
    if is_train: model.assign_lr(sess, lr)

    iters = 0
    costs = 0
    state = sess.run(model.init_state)

    batches = my_batch_producer(raw_data, config.batch_size, config.num_steps)

    for batch in batches:
        # run the model on current batch
        if is_train:
            _, c, state = sess.run(
                [model.train_op, model.cost, model.state],
                feed_dict={model.x: batch[0], model.y: batch[1],
                           model.init_state: state})
        else:
            c, state = sess.run([model.cost, model.state],
                                feed_dict={model.x: batch[0], model.y: batch[1],
                                           model.init_state: state})

        costs += c
        step = iters // config.num_steps
        if is_train and args.verbose == '1' \
                and step % (batches.epoch_size // 10) == 10:
            print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
            print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
            print('speed =',
                  round(iters * config.batch_size / (time.time() - start_time)),
                  'wps')
        iters += config.num_steps

    return np.exp(costs / iters)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.config == 'PTBSmall':
        config = PTBSmallConfig()
    elif args.config == 'PTBMedium':
        config = PTBMediumConfig()
    elif args.config == 'WT2Small':
        config = WT2SmallConfig()
    elif args.config == 'WT2Medium':
        config = WT2MediumConfig()
    else:
        sys.exit('Invalid config.')
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    train_raw_data, valid_raw_data, test_raw_data = read_data(args, config)

    with tf.variable_scope('Model', reuse=False, initializer=initializer):
        train = Train(config)
    print('Model size is: ', model_size())

    with tf.variable_scope('Model', reuse=True, initializer=initializer):
        valid = Model(config, need_reuse=True)

    with tf.variable_scope('Model', reuse=True, initializer=initializer):
        test = Model(config, need_reuse=True)

    saver = tf.train.Saver()

    if args.is_train == '1':
        num_epochs = config.num_epochs
        init = tf.global_variables_initializer()
        learning_rate = config.learning_rate

        with tf.Session() as sess:
            sess.run(init)

            saver.save(sess, os.path.join(
                args.save_dir, args.prefix + '-model.ckpt'))

            prev_valid_ppl = float('inf')
            best_valid_ppl = float('inf')

            for epoch in range(num_epochs):
                train_ppl = run_epoch(
                    sess, train, train_raw_data, config, is_train=True,
                    lr=learning_rate)
                print('epoch', epoch + 1, end=': ')
                print('train ppl = %.3f' % train_ppl, end=', ')
                print('lr = %.3f' % learning_rate, end=', ')

                # Get validation set perplexity
                valid_ppl = run_epoch(
                    sess, valid, valid_raw_data, config, is_train=False)
                print('valid ppl = %.3f' % valid_ppl)

                # Update the learning rate if necessary
                if valid_ppl >= best_valid_ppl:
                    learning_rate *= config.lr_decay

                # Save model if it gives better valid ppl
                if valid_ppl < best_valid_ppl:
                    save_path = saver.save(sess, os.path.join(
                        args.save_dir, args.prefix + '-model.ckpt'))
                    print('Valid ppl improved. Model saved in file: %s' % save_path)
                    best_valid_ppl = valid_ppl

                if valid_ppl > 2000 and epoch > 6:
                    sys.exit("ERROR!!!")

                if epoch > 20 and valid_ppl > 750:
                    sys.exit("WEAK MODEL!!!")

    # Evaluation of a trained model on test set
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(
            sess, os.path.join(args.save_dir, args.prefix + '-model.ckpt'))
        print('Model restored.')

        # Get test set perplexity
        test_ppl = run_epoch(
            sess, test, test_raw_data, config, is_train=False)
        print('Test set perplexity = %.3f' % test_ppl)

        with open("results/cosine_squasher_40_10.txt", "a") as f:
            f.write("Hidden " + str(config.hidden_size) + ", context: " + str(config.context_size) \
                    + ", emb_size: " + str(config.word_vocab_size) + ", lr: " + str(config.learning_rate) \
                    + ", decay: " + str(config.lr_decay) + ", num_steps: " + str(config.num_steps) \
                    + ", batch_size: " + str(config.batch_size) + ", grad_norm: " + str(config.max_grad_norm) \
                    + ", alpha: " + str(config.alpha) + ", max_epoch: " + str(config.num_epochs) + "init_Scale: " + str(
                config.init_scale))
            f.write("\n")
            f.write(
                "Train ppl " + str(train_ppl) + " Valid_ppl: " + str(valid_ppl) + "-------- TEST PPL: " + str(test_ppl))
            f.write("\n")
            f.write("\n")
    tf.reset_default_graph()