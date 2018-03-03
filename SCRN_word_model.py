import matplotlib
matplotlib.use('Agg')
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import collections
import numpy as np
import tensorflow as tf
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.7
import time
import sys
import pickle
import argparse
import copy
import csv
from tensorflow.python.ops import rnn_cell_impl
import matplotlib.pyplot as plt
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import dropoutwrapper_SCRNN as mydp
from tensorflow.python.ops import variable_scope as vs
from sklearn.metrics.pairwise import cosine_similarity

zaremba = False
small = False


class Config(object):

    print(zaremba)

    init_scale = 0.3
    
    learning_rate = 0.52
    lr_decay = 0.78
    max_grad_norm = 5
    num_layers = 2
    num_steps = 33
    hidden_size= emb_size = 210
    max_epoch = 10
    max_max_epoch = 35
    
    input_keep_prob= 0.96
    rnn_keep_prob = 0.81
    
    batch_size = 16
    vocab_size = 10000
    
    context_size = 40
    alpha = 0.95
    mom=0.51
    nesterov=False


        


    

def read_data(config):
    '''read data sets, construct all needed structures and update the config'''
    word_data = open('lstm/data/ptb.train.txt', 'r').read().replace('\n', ' <eos>').split()
    words = list(set(word_data))
                                                            
    word_data_size, word_vocab_size = len(word_data), len(words)
    print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
    config.word_vocab_size = word_vocab_size

    word_to_ix = {word: i for i, word in enumerate(words)}
    ix_to_word = {i: word for i, word in enumerate(words)}

    def get_word_raw_data(input_file):
        data = open(input_file, 'r').read().replace('\n', ' <eos>').split()
        return [word_to_ix[w] for w in data]

    train_raw_data = get_word_raw_data('lstm/data/ptb.train.txt')
    valid_raw_data = get_word_raw_data('lstm/data/ptb.valid.txt')
    test_raw_data = get_word_raw_data('lstm/data/ptb.test.txt')

    return train_raw_data, valid_raw_data, test_raw_data, word_to_ix, ix_to_word


class batch_producer(object):
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
        #self.k = 50
        
        #self.tr = tr
        
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
 


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """
        Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
        Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state and `h` is the output.
        Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
        return c.dtype

class custom_SCRNN(tf.contrib.rnn.RNNCell):     
    def __init__(self, batch_size, input_size,
                     hidden_size, context_size, alpha):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._context_size = context_size
        self._batch_size = batch_size
        self._alpha = alpha


    @property
    def state_size(self):
        #return self._hidden_size + self._context_size
        return LSTMStateTuple(self._context_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size+self._context_size

    def __call__(self, inputs, state,  scope=None):
        
        #hidden_state = tf.slice(state, begin=(0, 0), size=(self._batch_size, self._hidden_size))

        #context_state = tf.slice(state, begin=(0, self._hidden_size),  size=(self._batch_size, self._context_size))
        
        context_state, hidden_state = state

        with vs.variable_scope(scope or type(self).__name__):
            
            B = vs.get_variable('B_matrix', shape=[self._input_size, self._context_size])

            A = vs.get_variable('A_matrix', shape=[self._input_size, self._hidden_size])

            R = vs.get_variable('R_matrix', shape=[self._hidden_size, self._hidden_size])

            P = vs.get_variable('P_matrix', shape=[self._context_size, self._hidden_size])

            bias_term = vs.get_variable('Bias', shape=[self._hidden_size], initializer=tf.constant_initializer(0.0))            
            
            new_context = (1.0 - self._alpha) * tf.matmul(inputs, B) + self._alpha * context_state

            new_hidden = tf.nn.sigmoid(tf.matmul(new_context, P) + tf.matmul(inputs, A)+tf.matmul(hidden_state, R)+bias_term)

            new_output = array_ops.concat(values=[new_hidden, new_context], axis=1)
            
            #U = vs.get_variable('U_matrix', shape=[self._context_size, self._context_size])
            #V = vs.get_variable('V_matrix', shape=[self._hidden_size, self._context_size])
            #new_output = tf.add(tf.matmul(new_hidden, U), tf.matmul(new_context, V))
            
            new_state = LSTMStateTuple(new_context, new_hidden)

        return  new_output, new_state


class my_model(object):
    
    def __init__(self, is_training, config):
       
        # get hyperparameters
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        init_scale = config.init_scale
        word_emb_dim = hidden_size = config.hidden_size
        size = config.hidden_size
        vocab_size = config.vocab_size
        context_size = config.context_size
        alpha = config.alpha
        emb_size = config.emb_size
        
        # placeholders for training data and labels
        self._x = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._y = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, emb_size])
            self._emb = embedding
            inputs = tf.nn.embedding_lookup(embedding, self._x)
        

        if is_training : inputs = tf.nn.dropout(inputs, config.input_keep_prob)
        inputs = tf.unstack(inputs, axis=1)
        

  
        #cell1 = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)
        
        cell1 = custom_SCRNN(batch_size, emb_size, hidden_size, context_size, alpha)
        #if is_training : cell1 = mydp.DropoutWrapper(cell1, context_size, hidden_size, batch_size,  output_keep_prob=config.rnn_keep_prob)
        if is_training : cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=config.rnn_keep_prob)

        
        #cell2 = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=not is_training)
        
        cell2 = custom_SCRNN(batch_size, cell1.output_size, hidden_size, context_size, alpha)
            #if is_training : cell2 = mydp.DropoutWrapper(cell2, context_size, hidden_size, batch_size, output_keep_prob=config.rnn_keep_prob)
        if is_training : cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=config.rnn_keep_prob)


        """ 

            cell1 = custom_SCRNN(batch_size, emb_size, hidden_size, context_size, alpha)
            #if is_training : cell1 = mydp.DropoutWrapper(cell1, context_size, hidden_size, batch_size,  output_keep_prob=config.rnn_keep_prob)
            if is_training : cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=config.rnn_keep_prob)

            cell2 = custom_SCRNN(batch_size, cell1.output_size, hidden_size, context_size, alpha)
            #if is_training : cell2 = mydp.DropoutWrapper(cell2, context_size, hidden_size, batch_size, output_keep_prob=config.rnn_keep_prob)
            if is_training : cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=config.rnn_keep_prob)
            
        """
       
        
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2], state_is_tuple = True)

        self._init_state = cell.zero_state(batch_size, tf.float32)
    
        output_size = cell.output_size
        
        
        outputs, state = tf.nn.static_rnn(
                            cell,
                            inputs,
                            dtype=tf.float32,
                            initial_state=self._init_state)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, output_size])
        
        softmax_w = tf.get_variable("softmax_w", [output_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._y,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = state
                
        if not is_training:
              return
   
        # training
        self._lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)        
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.MomentumOptimizer(self.lr, config.mom, use_nesterov=config.nesterov)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
        
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
                                        
    @property
    def drop_mat(self):
        return self._drop_mat                                       

    @property
    def init_state(self):
        return self._init_state

    @property
    def cost(self):
        return self._cost
    
    @property
    def embedding(self):
        return self._emb

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    


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


def run_epoch(sess, model, raw_data, eval_op, config, is_train=False):
    start_time = time.time()

    iters = 0
    costs = 0
    state = sess.run(model.init_state)
    #state = model.init_state.eval()
    

    batches = batch_producer(raw_data, config.batch_size, config.num_steps)
  
    
    for (batch_x, batch_y) in batches:
        # run the model on current batch  
        _, c, state = sess.run(
                [eval_op, model.cost, model.final_state],
                feed_dict={model.x: batch_x, 
                           model.y: batch_y,
                           model.init_state: state})


        costs += c
        step = iters // config.num_steps

        if is_train and step % (batches.epoch_size // 10) == 10:
            print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
            print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
            print('speed =',
                  round(iters * config.batch_size / (time.time() - start_time)),
                  'wps')
        iters += config.num_steps

    return np.exp(costs / iters)


def main(_):
    
    config = Config()
    eval_config = Config()
    #eval_config.batch_size = 1
    #eval_config.num_steps = 1
    
    train_raw_data, valid_raw_data, test_raw_data, word_to_ix, ix_to_word = read_data(config)
    
    print('Model size is: ', model_size())
    print(config.vocab_size)


    num_epochs = config.max_max_epoch
    init = tf.global_variables_initializer()
    #learning_rate = config.learning_rate
    
    with tf.Graph().as_default(), tf.Session(config=conf) as sess:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = my_model(is_training=True, config=config)
        
        print('Model size is: ', model_size())
        saver = tf.train.Saver()
        
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = my_model(is_training=False, config=config)
            mtest = my_model(is_training=False, config=eval_config)  
        
        tf.global_variables_initializer().run()
        
        prev_valid_ppl = float('inf')
        best_valid_ppl = float('inf')
        valid_ppl = 0
        lr_rate = config.learning_rate
        
        valids = []
        trains = []
        epochs = []
        model.assign_lr(sess, lr_rate)
        for epoch in range(num_epochs):
            
            
            
            
          
            lr_decay = config.lr_decay ** max(epoch + 1 - config.max_epoch, 0.0)
            model.assign_lr(sess, config.learning_rate * lr_decay)           
            lr_rate = config.learning_rate * lr_decay
          
            
            train_ppl = run_epoch(
                sess, model, train_raw_data, model.train_op, config, is_train=True)
            print('epoch', epoch + 1, end=': ')
            print('train ppl = %.3f' % train_ppl, end=', ')
            print('lr = %.3f' % lr_rate, end=', ')

            # Get validation set perplexity
            valid_ppl = run_epoch(
                sess, mvalid, valid_raw_data, tf.no_op(), config, is_train=False)
            print('valid ppl = %.3f' % valid_ppl)

            """
            if best_valid_ppl-valid_ppl>1:
                model.assign_lr(sess, lr_rate)
            else:
                lr_rate =  lr_rate*config.lr_decay
                model.assign_lr(sess, lr_rate)
            """
                
            # Save model if it gives better valid ppl
            if valid_ppl < best_valid_ppl:
                best_valid_ppl = valid_ppl          
                
                save_path = saver.save(sess, 'PTB/saves2/LSTM_LARGE_PTB/model.ckpt')
                print('Valid ppl improved. Model saved in file: %s' % save_path)
                
            
            
            valids.append(valid_ppl)
            trains.append(train_ppl)
            epochs.append(epoch)
            if lr_rate == 0:
                break 
    
        '''Evaluation of a trained model on test set'''    
        # Restore variables from disk.
        saver.restore(sess, 'PTB/saves2/LSTM_LARGE_PTB/model.ckpt')
        #with open('PTB/saves2/LSTM_LARGE_PTB_WORD/lstm_word.txt', 'w') as f: 
        
        """
        matrix = sess.run(model.embedding)


        id1 = (word_to_ix['woman'])
        id2 = (word_to_ix['king'])
        id3 = (word_to_ix['me'])
        id4 = (word_to_ix['we'])
        id5 = (word_to_ix['say'])
        id6 = (word_to_ix['good'])
        id7 = (word_to_ix['salary'])
        id8 = (word_to_ix['study'])
        id9 = (word_to_ix['tiger'])
        id10 = (word_to_ix['joke'])
        ids = [id1,id2,id3,id4,id5,id6,id7,id8,id9,id10]

        with open('PTB/saves2/LSTM_LARGE_PTB_WORD/lstm_word.txt', 'w') as f:
            for ix in ids:
                similars = cosine_similarity(matrix[ix], matrix)
                similars = np.array(similars[0])
                best = similars.argsort()[-6:][::-1]
                f.write('_____________')
                f.write(ix_to_word[ix])

                for i in best:
                    print(ix_to_word[i])
                    f.write(ix_to_word[i]+'\n')
        
        """
        print('Model restored.')

        # Get test set perplexity
        test_ppl = run_epoch(sess, mtest, test_raw_data, tf.no_op(), config, is_train=False)
        print('Test set perplexity = %.3f' % test_ppl)
        
        
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, valids, label = 'valids')
        plt.plot(epochs, trains, label = 'train')
        plt.plot(epochs[-1], test_ppl, label = 'test')
        plt.legend(loc='best')
        plt.title("LR vs PPL")
        out_png = 'PTB/out_file.png'
        plt.savefig(out_png, dpi=150)
        plt.show(block=True) 
        
        with open("PTB/results_SCRNN_dropout.txt", "a") as f:
            #writer = csv.writer(f)
            f.write("Hidden "+str(config.hidden_size)+", context: "+ str(config.context_size)+", emb_size: "+ str(config.emb_size)+", lr: "+str(config.learning_rate)+", decay: "+ str(config.lr_decay)+", max_epoch: "+ str(config.max_epoch)
                    +", dropout: "+str(config.rnn_keep_prob)+", num_steps: "+str(config.num_steps)+", batch_size: "+str(config.batch_size)+", mom: "+str(config.mom)+", grad_norm: "+str(config.max_grad_norm)+", Model_Size: "+str(model_size())+", init_Scale: "+str(config.init_scale)+", nesterov: "+str(config.nesterov))
            f.write("\n")
            f.write("Train ppl "+str(train_ppl)+" Valid_ppl: "+str(valid_ppl)+"-------- TEST PPL: "+str(test_ppl))
            f.write("\n")
            f.write("\n")
            
if __name__ == "__main__":
    tf.app.run()