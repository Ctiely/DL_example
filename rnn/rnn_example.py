import tensorflow as tf
import numpy as np
import os
import math

from tqdm import tqdm
from tensorboardX import SummaryWriter


def generator(dataset, data_lengths, batch_size, words_idx):
    n_sample = len(dataset)
    indexs = np.arange(n_sample)
    np.random.shuffle(indexs)
    batch_datas = []
    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
        span_index = indexs[span_index]
        batch_lengths = data_lengths[span_index]
        batch_words = np.zeros((len(batch_lengths), batch_lengths.max()), dtype=np.int32)
        batch_targets = np.zeros((len(batch_lengths), batch_lengths.max()), dtype=np.int32)
        for i, index in enumerate(span_index):
            cur_data = dataset[index]
            length = len(cur_data)
            embedding = np.array([words_idx[c] for c in cur_data])
            batch_words[i, : length] = embedding
            batch_targets[i, : length - 1], batch_targets[i, length - 1] = embedding[1:], len(words_idx) - 1
        batch_datas.append((batch_words, batch_targets, batch_lengths))
    return batch_datas


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class RNNModel(object):
    def __init__(self, vocab_size, hidden_size,
                 keep_prob=0.5,
                 batch_size=64,
                 max_grad_norm=1.0,
                 embedding_size=100,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 save_path="./rnn_example"):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.training_batchsize = batch_size
        self.lr_schedule = lr_schedule
        self.save_path = save_path
        
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)
        self.sw = SummaryWriter(log_dir=self.save_path)
        
        self._build_model()
        self._build_algorithm()
        self._prepare()
        
    def _build_model(self):
        self.input_words = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')
        self.keep_prob_hd = tf.placeholder(tf.float32)
        batch_size = tf.shape(self.input_words)[0]
        max_length = tf.shape(self.input_words)[1]
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
            self.gru_inputs = tf.nn.embedding_lookup(embedding, self.input_words)
        
        self.mask = tf.sequence_mask(self.lengths, max_length)
        gru = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        dropout = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=self.keep_prob_hd)
        self.initial_state = dropout.zero_state(batch_size, tf.float32)
        self.gru_outputs, self.final_state = tf.nn.dynamic_rnn(
                dropout, self.gru_inputs,
                initial_state=self.initial_state,
                sequence_length=self.lengths
                )
        
        mask_outputs = tf.boolean_mask(self.gru_outputs, self.mask)
        self.outputs = tf.layers.dense(
                mask_outputs, self.vocab_size,
                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01)
                )
    
    def _build_algorithm(self):
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)
        self.pred_prob = tf.nn.softmax(self.outputs)
        
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        mask_targets = tf.boolean_mask(self.targets, self.mask)
        self.total_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=mask_targets, logits=self.outputs
                        ),
                axis=0
                )
        grads = tf.gradients(self.total_loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())                                
    
    def _prepare(self):
        self.saver = tf.train.Saver(max_to_keep=10)

        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.sess.run(tf.global_variables_initializer())
        self.load_model()
        
    def save_model(self):
        """Save model to `save_path`."""
        save_dir = os.path.join(self.save_path, "model")
        os.makedirs(save_dir, exist_ok=True)
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(save_dir, "model"),
            global_step,
            write_meta_graph=True
        )

    def load_model(self):
        """Load model from `save_path` if there exists."""
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.save_path, "model"))
        if latest_checkpoint:
            print("## Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("## New start!")
    
    def update(self, dataset, data_lengths, words_idx, update_ratio):
        batch_datas = generator(
                dataset, data_lengths, self.training_batchsize, words_idx
                )
        loss = 0
        step = 0
        for mini_words, mini_targets, mini_lengths in tqdm(batch_datas):
            step += 1
            fd = {
                self.input_words: mini_words,
                self.targets: mini_targets,
                self.lengths: mini_lengths,
                self.keep_prob_hd: self.keep_prob,
                self.moved_lr: self.lr_schedule(update_ratio)
                }

            cur_loss, _ = self.sess.run(
                [self.total_loss, self.train_op],
                feed_dict=fd
                )
            loss += cur_loss
            global_step = self.sess.run(tf.train.get_global_step())
            self.sw.add_scalar(
                'loss',
                cur_loss,
                global_step=global_step)
        return loss / step
    
    def sample(self, n_samples, prime, idx_words):
        samples = prime[:]
        state = self.sess.run(self.initial_state,
                              feed_dict={self.input_words: [[0]]})
        preds = np.ones((self.vocab_size, ))
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = c
            fd = {self.input_words: x,
                  self.lengths: [1],
                  self.keep_prob_hd: 1.0,
                  self.initial_state: state}
            preds, state = self.sess.run([self.pred_prob, self.final_state],
                                         feed_dict=fd)

        c = pick_top_n(preds, self.vocab_size)
        samples.append(c)

        for i in range(n_samples):
            if c == self.vocab_size - 1:
                break
            x = np.zeros((1, 1))
            x[0, 0] = c
            fd = {self.input_words: x,
                  self.lengths: [1],
                  self.keep_prob_hd: 1.0,
                  self.initial_state: state}
            preds, state = self.sess.run([self.pred_prob, self.final_state],
                                         feed_dict=fd)

            c = pick_top_n(preds, self.vocab_size)
            samples.append(c)

        return ''.join([idx_words[c] for c in samples])


if __name__ == "__main__":
    total_updates = 100
    save_model_freq = 20
    hidden_size = 128
    
    with open('data/poetry.txt', 'r') as f:
        texts = f.readlines()
    
    total_words = set()
    for text in texts:
        total_words |= set(text)
    
    total_words = list(total_words)
    total_words.append('#')
    vocab_size = len(total_words)
    words_idx = dict(zip(total_words, range(len(total_words))))
    idx_words = dict(zip(range(len(total_words)), total_words))
    dataset = [text.strip() for text in texts]
    data_lengths = np.array(list(map(len, dataset)))
    
    rnn = RNNModel(vocab_size, hidden_size)
    
    for epoch in range(total_updates):
        epoch += 1
        loss = rnn.update(dataset, data_lengths, words_idx,
                          min(0.9, epoch / total_updates))
        print(f'\n>>>>Train Loss: {loss}')
        if epoch % save_model_freq == 0:
            rnn.save_model()
    
    print(rnn.sample(10, [np.random.randint(vocab_size)], idx_words))

