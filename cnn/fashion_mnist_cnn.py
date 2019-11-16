import tensorflow as tf
import skimage
import math
import os
import numpy as np
import random

from collections import defaultdict
from tqdm import tqdm
from tensorboardX import SummaryWriter


def generator(data_batch, batch_size):
    n_sample = data_batch[0].shape[0]
    index = np.arange(n_sample)
    np.random.shuffle(index)
    batch_datas = []
    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
        span_index = index[span_index]
        batch_datas.append([x[span_index, :] if x.ndim > 1 else x[span_index] for x in data_batch])
    return batch_datas


class DataSet(object):
    def __init__(self, data_file):
        imgs = []
        labels = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                img_path, label = line.strip().split()
                imgs.append(skimage.io.imread(img_path))
                labels.append(int(label))
        self.imgs = np.asarray(imgs)
        self.labels = np.asarray(labels)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


class CNNModel(object):
    def __init__(self, img_size, output_size,
                 batch_size=64,
                 max_grad_norm=1.0,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 save_path="./fashion_mnist_cnn"):
        self.img_size = img_size
        self.output_size = output_size
        self.training_batchsize = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.save_path = save_path
        
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)
        self.sw = SummaryWriter(log_dir=self.save_path)
        
        self._build_model()
        self._build_algorithm()
        self._prepare()
    
    def _prepare(self):
        self.saver = tf.train.Saver(max_to_keep=10)
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)
        self.sess.run(tf.global_variables_initializer())
        self.load_model()
    
    def _build_model(self):
        self.imgs = tf.placeholder(
                tf.uint8, [None, *self.img_size]
                )
        
        self.imgs_ = tf.divide(tf.cast(self.imgs, tf.float32), 255.0)
        self.conv1 = tf.layers.conv2d(
                inputs=self.imgs_,
                filters=32,
                kernel_size=3,
                strides=1,
                activation=tf.nn.relu
                )
        
        self.max_pool1 = tf.layers.max_pooling2d(
                inputs=self.conv1,
                pool_size=2,
                strides=2
                )
        
        self.conv2 = tf.layers.conv2d(
                inputs=self.max_pool1,
                filters=64,
                kernel_size=3,
                strides=1,
                activation=tf.nn.relu
                )
        
        self.max_pool2 = tf.layers.max_pooling2d(
                inputs=self.conv2,
                pool_size=2,
                strides=2
                )
        
        self.conv3 = tf.layers.conv2d(
                inputs=self.max_pool2,
                filters=128,
                kernel_size=3,
                strides=1,
                activation=tf.nn.relu
                )
        
        self.flatten = tf.contrib.layers.flatten(self.conv3)
        self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.relu
                )
        self.outputs = tf.layers.dense(
                inputs=self.fc,
                units=self.output_size
                )
        
    def _build_algorithm(self):
        self.labels = tf.placeholder(tf.int32)
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)
        self.preds = tf.argmax(self.outputs, axis=1)
        
        self.total_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.labels, logits=self.outputs
                        ),
                axis=0
                )
        grads = tf.gradients(self.total_loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # train_op
        self.train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())
        
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
    
    def update(self, train_imgs, train_labels, update_ratio):
        batch_datas = generator(
                [train_imgs, train_labels], batch_size=self.training_batchsize
                )
        
        loss = 0
        accuracy = 0
        step = 0
        for minibatch_imgs, minibatch_targets in tqdm(batch_datas):
            step += 1
            fd = {
                self.imgs: minibatch_imgs[:, :, :, np.newaxis],
                self.labels: minibatch_targets,
                self.moved_lr: self.lr_schedule(update_ratio)
                }

            batch_loss, preds, _ = self.sess.run(
                [self.total_loss, self.preds, self.train_op],
                feed_dict=fd
                )
            batch_accuracy = np.mean(preds == minibatch_targets)

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
    
    def predict(self, imgs):
        imgs = np.asarray(imgs)
        if imgs.ndim == 3:
            imgs = imgs[:, :, :, np.newaxis]
        elif imgs.ndim == 2:
            imgs = imgs[np.newaxis, :, :, np.newaxis]
        preds = self.sess.run(self.preds, feed_dict={self.imgs: imgs})
        return preds

    
if __name__ == '__main__':
    num_class = 2000
    train_data = DataSet('data/train.txt')
    test_data = DataSet('data/test.txt')
    
    groups = defaultdict(list)
    for i in range(len(train_data)):
        groups[train_data.labels[i]].append(i)
    
    random.seed(0)
    selected = []
    for group in groups:
        selected.extend(random.sample(groups[group], num_class))
    
    imgs = train_data.imgs[selected]
    labels = train_data.labels[selected]
    
    total_updates = 1000
    save_model_freq = 20
    
    cnn = CNNModel((28, 28, 1), 10)
    
    epoch = 0
    while True:
        epoch += 1
        loss, accuracy = cnn.update(imgs, labels, min(0.9, epoch / total_updates))
        print(f'\n>>>>Traine poch: {epoch}, Loss: {loss}, Accuracy: {accuracy}')
        if epoch % save_model_freq == 0:
            cnn.save_model()
            accuracy = np.mean(cnn.predict(test_data.imgs) == test_data.labels)
            print(f'\nTest accuracy: {accuracy}')

