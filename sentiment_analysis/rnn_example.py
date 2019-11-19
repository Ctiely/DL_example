import tensorflow as tf
import numpy as np
import os
import math
import pandas as pd
# import re

from collections import defaultdict
from tqdm import tqdm
from tensorboardX import SummaryWriter
# from bs4 import BeautifulSoup
# from nltk.corpus import stopwords


def generator(dataset, target, data_lengths, batch_size, words_idx):
    n_sample = len(dataset)
    indexs = np.arange(n_sample)
    np.random.shuffle(indexs)
    batch_datas = []
    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
        span_index = indexs[span_index]
        batch_lengths = data_lengths[span_index]
        batch_target = target[span_index]
        batch_words = np.zeros((len(batch_lengths), batch_lengths.max()), dtype=np.int32)
        for i, index in enumerate(span_index):
            cur_data = dataset[index]
            length = len(cur_data)
            batch_words[i, : length] = np.array([words_idx[c] for c in cur_data])
        batch_datas.append((batch_words, batch_target, batch_lengths))
    return batch_datas


class RNNModel(object):
    def __init__(self, num_class, vocab_size, hidden_size,
                 batch_size=64,
                 max_grad_norm=1.0,
                 embedding_size=100,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 save_path="./rnn_example"):
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding_size = embedding_size
        self.max_grad_norm = max_grad_norm
        self.training_batchsize = batch_size
        self.lr_schedule = lr_schedule
        self.save_path = save_path
        
        self.training = None
        
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
        batch_size = tf.shape(self.input_words)[0]
        max_length = tf.shape(self.input_words)[1]
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                    'embedding', [self.vocab_size, self.embedding_size])
            self.gru_inputs = tf.nn.embedding_lookup(embedding, self.input_words)
        
        gru = tf.contrib.cudnn_rnn.CudnnGRU(1, self.hidden_size)
        self.gru_outputs, _ = gru(inputs=self.gru_inputs, training=self.training)
        indexs = tf.range(0, batch_size) * max_length + self.lengths - 1
        outputs = tf.gather(
                tf.reshape(self.gru_outputs, [-1, self.hidden_size]),
                indexs
                )
        self.outputs = tf.layers.dense(
                outputs, self.num_class,
                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01)
                )
    
    def _build_algorithm(self):
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)
        self.preds = tf.argmax(self.outputs, axis=1)
        
        self.targets = tf.placeholder(tf.int32, name='targets')
        self.total_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.targets, logits=self.outputs
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
    
    def update(self, dataset, target, data_lengths, words_idx, update_ratio):
        self.training = True
        batch_datas = generator(
                dataset, target, data_lengths, self.training_batchsize, words_idx
                )
        loss = 0
        accuracy = 0
        step = 0
        for mini_words, mini_targets, mini_lengths in tqdm(batch_datas):
            step += 1
            fd = {
                self.input_words: mini_words,
                self.targets: mini_targets,
                self.lengths: mini_lengths,
                self.moved_lr: self.lr_schedule(update_ratio)
                }

            batch_loss, preds, _ = self.sess.run(
                [self.total_loss, self.preds, self.train_op],
                feed_dict=fd
                )
            batch_accuracy = np.mean(preds == mini_targets)

            global_step = self.sess.run(tf.train.get_global_step())
            self.sw.add_scalar(
                'accuracy',
                batch_accuracy,
                global_step=global_step)
            self.sw.add_scalar(
                'loss',
                batch_loss,
                global_step=global_step)
            loss += batch_loss
            accuracy += batch_accuracy
        return loss / step, accuracy / step

    def predict(self, batch_data, batch_length):
        self.training = False
        preds = self.sess.run(self.preds,
                              feed_dict={self.input_words: batch_data,
                                         self.lengths: batch_length})
        return preds


if __name__ == '__main__':
    '''
    datas = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    
    def review_to_words(raw_review):
        review_text = BeautifulSoup(raw_review, "lxml").get_text() 
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
        words = letters_only.lower().split()                             
        stops = set(stopwords.words("english"))                  
        meaningful_words = [w for w in words if not w in stops]
        return ' '.join(meaningful_words)
    
    total_datas = pd.DataFrame()
    total_datas['review'] = datas['review'].apply(review_to_words)
    total_datas['target'] = datas['sentiment']
    total_datas.to_csv('data/movies.csv', index=False)
    '''
    total_datas = pd.read_csv("data/movies.csv", header=0)
    
    words_list = total_datas['review'].apply(lambda x: x.split())
    words_freq = defaultdict(int)
    for i in range(len(total_datas)):
        for word in words_list[i]:
            words_freq[word] += 1
    
    datas = []
    for i in range(len(total_datas)):
        data = []
        for word in words_list[i]:
            if words_freq[word] >= 30:
                data.append(word)
        datas.append(data)
    total_datas['review'] = datas
    
    total_words = set()
    for i in range(len(datas)):
        total_words |= set(datas[i])
    vocab_size = len(total_words)
    words_idx = dict(zip(total_words, range(len(total_words))))
    idx_words = dict(zip(range(len(total_words)), total_words))
    
    positive_indexs = np.where(total_datas['target'].values == 1)[0]
    negative_indexs = np.where(total_datas['target'].values == 0)[0]
    
    train_ratio = 0.7
    np.random.seed(0)
    selected = np.random.choice(positive_indexs,
                                size=int(train_ratio * len(total_datas) / 2),
                                replace=False).tolist() + \
               np.random.choice(negative_indexs,
                                size=int(train_ratio * len(total_datas) / 2),
                                replace=False).tolist()
    selected.sort()
    indexs = np.zeros(len(total_datas), dtype=np.bool)
    indexs[selected] = True
    train_data = total_datas[indexs]
    test_data = total_datas[~indexs]
    
    dataset = train_data['review'].tolist()
    data_lengths = np.array(list(map(len, dataset)))
    
    test_dataset = test_data['review'].tolist()
    test_data_lengths = np.array(list(map(len, test_dataset)))
    batch_test_datas = generator(
        test_dataset, test_data['target'].values, test_data_lengths, 64, words_idx
        )
    
    total_updates = 1000
    save_model_freq = 10
    eval_step = 5
    hidden_size = 128
    
    rnn = RNNModel(2, vocab_size, hidden_size)
    
    epoch = 0
    while True:
        epoch += 1
        loss, accuracy = rnn.update(
                dataset, train_data['target'].values, data_lengths,
                words_idx, min(0.9, epoch / total_updates)
                )
        print(f'>>>>Traine poch: {epoch}, Loss: {loss}, Accuracy: {accuracy}')
        if epoch % save_model_freq == 0:
            rnn.save_model()
        
        if epoch % eval_step == 0:
            accuracy = 0
            for batch_data, batch_target, batch_length in tqdm(batch_test_datas):
                accuracy += np.sum(rnn.predict(batch_data, batch_length) == batch_target)
            print(f'Test accuracy: {accuracy / len(test_data)}')
            rnn.sw.add_scalar(
                    'test_accuracy',
                    accuracy,
                    global_step=epoch // eval_step)

