# -*- utf-8 -*-
import tensorflow as tf
import data_tools as dt
import numpy as np
import os
import time
import logging
logger = logging.getLogger('main.lstm_model')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LSTModel:
    def __init__(self, class_num,  lstm_dims,
                  fc_size= 125, max_sent_len= 30):
        self.class_num = class_num
        self.lstm_dims = lstm_dims
        self.fc_size = fc_size
        self.max_sent_len = max_sent_len

        # ----placeholder
        self.learning_rate = tf.placeholder_with_default(0.01, shape=(), name='learning_rate')      # 学习率
        self.keep_prob = tf.placeholder_with_default(
            1.0, shape=(), name='keep_prob')           # dropout keep probability
        self.l2reg = tf.placeholder_with_default(0.0, shape=(), name='L2reg')               # L2正则化参数
        self.ATTENTION_SIZE =50
    def inputs_layer(self):
        with tf.name_scope('input_layer'):
            self.inputs = tf.placeholder(tf.float32, [None, self.max_sent_len], name='inputs')  # 输入数据x placeholder
            self.labels = tf.placeholder(tf.int32, [None, self.class_num], name='labels')  # 输入数据y placeholder
        return self.inputs


    def lstm_layer(self, inputs):
        inputs = tf.expand_dims(inputs, axis=2)
        with tf.name_scope("lstm_layer"):
            #embed = tf.nn.dropout(inputs, keep_prob=self.keep_prob)  # dropout
            lstms_l = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_dims]
            lstms_r = [tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_dims]
            drops_l = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_l]
            drops_r = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.keep_prob) for lstm in lstms_r]
            cell_l = tf.contrib.rnn.MultiRNNCell(drops_l)
            cell_r = tf.contrib.rnn.MultiRNNCell(drops_r)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_l,
                cell_r,
                inputs=inputs,
                dtype=tf.float32,
            )

            lstm_outputs = tf.concat(outputs, -1)
            outputs = lstm_outputs[:, -1]

        return outputs



    def fc_layer(self, inputs):
        # initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope("fc_layer"):
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='drop_out')  # dropout
            # outputs = tf.contrib.layers.fully_connected(inputs, self.fc_size, activation_fn=tf.nn.relu)
            outputs = tf.layers.dense(inputs, self.fc_size, activation=tf.nn.relu)
        return outputs

    def output_layer(self, inputs):
        "output layer"
        with tf.name_scope("output_layer"):
            inputs = tf.layers.dropout(inputs, rate=1-self.keep_prob)
            outputs = tf.layers.dense(inputs, self.class_num, activation=None)
            # outputs = tf.contrib.layers.fully_connected(inputs, self.class_num, activation_fn=None)
        return outputs

    def set_loss(self):
        "softmax"
        with tf.name_scope("loss_scope"):
            reg_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(self.l2reg),
                tf.trainable_variables()
            )
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
               logits=self.predictions, labels=self.labels)) + reg_loss
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=self.predictions, labels=self.labels))

    def set_accuracy(self):
        "accuracy"
        with tf.name_scope("accuracy_scope"):
            correct_pred = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # ---GLOBAL---准确率

    def set_optimizer(self):
        "optimizer"
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def build(self):
        "LSTM Model building"
        inputs = self.inputs_layer()
        lstm_outputs = self.lstm_layer(inputs)
        #attention_output = self.attention_layer(lstm_outputs)
        fc_outputs = self.fc_layer(lstm_outputs)
        self.predictions = self.output_layer(fc_outputs)
        self.set_loss()
        self.set_optimizer()
        self.set_accuracy()


def train(lstm_model, learning_rate, train_x, train_y, dev_x, dev_y, max_epochs, batch_size, keep_prob, l2reg,
          show_step=10, checkpoint_path="./checkpoints", model_name=None, no_improve=5):
    # save best model
    if model_name is None:
        model_name = str(time.time()).replace('.', '')[:11]
    best_model_path = checkpoint_path + '/best/' + model_name
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Train Summaries
        train_loss = tf.summary.scalar("train_loss", lstm_model.loss)
        train_acc = tf.summary.scalar("train_acc", lstm_model.accuracy)
        train_summary_op = tf.summary.merge([train_loss, train_acc])
        train_summary_writer = tf.summary.FileWriter('./log/train', sess.graph)

        # Dev summary writer
        dev_summary_writer = tf.summary.FileWriter('./log/dev', sess.graph)
        sess.run(tf.global_variables_initializer())
        n_batches = len(train_x)//batch_size
        step = 0
        best_dev_acc = 0
        min_dev_loss = float('inf')

        no_improve_num = 0
        train_sum=[]
        for e in range(max_epochs):
            for id_, (x, y) in enumerate(dt.make_batches(train_x, train_y, batch_size), 1):
                step += 1
                feed = {
                    lstm_model.inputs: x,
                    lstm_model.labels: y,
                    lstm_model.learning_rate: learning_rate,
                    lstm_model.keep_prob: keep_prob,
                    lstm_model.l2reg: l2reg
                }
                train_loss, _, train_acc, train_summary = sess.run(
                    [lstm_model.loss, lstm_model.optimizer, lstm_model.accuracy, train_summary_op],
                    feed_dict=feed,
                    # options=run_options,                                  # for meta 日志 - **3
                    # run_metadata=run_metadata                             # for meta 日志 - **4
                )
                train_sum.append(train_acc)

                train_summary_writer.add_summary(train_summary, step)  # 写入日志

                if show_step > 0 and step % show_step == 0:
                    info = "Epoch {}/{} ".format(e+1, max_epochs) + \
                        " - Batch {}/{} ".format(id_+1, n_batches) + \
                        " - Loss {:.5f} ".format(train_loss) + \
                        " - Acc {:.5f}".format(train_acc)
                    logger.info(info)

            # epoch test
            dev_acc_s = []
            dev_loss_s = []
            for xx, yy in dt.make_batches(dev_x, dev_y, batch_size):
                feed = {
                    lstm_model.inputs: xx,
                    lstm_model.labels: yy,
                    lstm_model.keep_prob: 1,
                }
                dev_batch_loss, dev_batch_acc = sess.run([lstm_model.loss, lstm_model.accuracy], feed_dict=feed)
                dev_acc_s.append(dev_batch_acc)
                dev_loss_s.append(dev_batch_loss)

            dev_acc = np.mean(dev_acc_s)    # dev acc
            dev_loss = np.mean(dev_loss_s)  # dev loss

          #dev log
            dev_summary = tf.Summary()
            dev_summary.value.add(tag="dev_loss", simple_value=dev_loss)
            dev_summary.value.add(tag="dev_acc", simple_value=dev_acc)
            dev_summary_writer.add_summary(dev_summary, step)

            info = "|Epoch {}/{}\t".format(e+1, max_epochs) + \
                "|Train-Loss| {:.5f}\t".format(train_loss) + \
                "|Dev-Loss| {:.5f}\t".format(dev_loss) + \
                "|Train-Acc| {:.5f}\t".format(np.mean(train_sum)) + \
                "|Dev-Acc| {:.5f}".format(dev_acc)
            logger.info(info)

            # save best model
            if best_dev_acc < dev_acc:
                best_dev_acc = dev_acc
                saver.save(sess, best_model_path + "/best_model.ckpt")
            #
            # saver.save(sess, best_model_path + "model.ckpt", global_step=e)

            # dev_loss
            if min_dev_loss > dev_loss:
                min_dev_loss = dev_loss
                no_improve_num = 0
            else:
                no_improve_num += 1

            # early stop
            if no_improve_num == no_improve:
                break

        logger.info("** The best dev accuracy: {:.5f}".format(best_dev_acc))


    return min_dev_loss


def test(lstm_model, test_x, test_y, batch_size, model_dir="./checkpoints/best"):

    best_folder = max([d for d in os.listdir(model_dir) if d.isdigit()])
    best_model_dir = model_dir + '/' + best_folder
    saver = tf.train.Saver()
    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(best_model_dir))
        for _, (x, y) in enumerate(dt.make_batches(test_x, test_y, batch_size), 1):
            feed = {
                lstm_model.inputs: x,
                lstm_model.labels: y,
                lstm_model.keep_prob: 1,
            }
            batch_acc = sess.run([lstm_model.accuracy], feed_dict=feed)
            test_acc.append(batch_acc)
        logger.info("** Test Accuracy: {:.5f}".format(np.mean(test_acc)))
