# -*- utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import data_tools as tools
import lstm as model_input
import data_preprocessing as dp
import numpy as np
import logging
from sklearn.preprocessing import LabelBinarizer


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logger = logging.getLogger('main')

# ================== define hyperpapermeter =================
learning_rate = 0.001    # learning rate
batch_size = 512       # mini-batch
keep_prob = 1          # drop out
l2reg = 0              # l2
lstm_sizes = [256]       # lstm dimention
fc_size = 256            # layer_size
max_epochs = 50


# ---- 其他参数
max_sent_len = 30
class_num = 2
show_step = 20
data_path = './data/'

# ================== data prepare =================
#loading data from data_preprocesssing
X_final,y_final, X_final_dev,y_final_dev, X_final_test, y_final_test = dp.load_data(data_path + "creditcard.csv")
# print(len(train_x))

encoder = LabelBinarizer()

one_hot_train_y = encoder.fit_transform(y_final)
one_hot_dev_y = encoder.fit_transform(y_final_dev)
one_hot_final_test_y = encoder.fit_transform(y_final_test)
# one_hot_train_y = list(one_hot_train_y)
# print(len(one_hot_train_y))
# print(type(one_hot_train_y))
# print(one_hot_train_y)


change_train_y =[]
change_dev_y =[]
change_final_test_y =[]
for item in one_hot_train_y:
    #a.repeat(5)
    # print(item.repeat(1))
    change_train_y.append(item.repeat(2))
for item in one_hot_dev_y:
    change_dev_y.append(item.repeat(2))
for item in one_hot_final_test_y:
    change_final_test_y.append(item.repeat(2))
change_train_y = np.array(change_train_y)
change_dev_y = np.array(change_dev_y)
change_final_test_y = np.array(change_final_test_y)
# print(change_train_y.type)
# exit()
train_x = np.array(X_final)
dev_x = np.array(X_final_dev)
train_y = np.array(change_train_y)
dev_y = np.array(y_final_dev)
X_final_test = np.array(X_final_test)
change_final_test_y = np.array(change_final_test_y)
# val_x = test_x
# val_y = test_y
test_x = X_final_test
test_y = change_final_test_y
val_x = dev_x
val_y = change_dev_y

model = model_input.LSTModel(class_num=class_num,  lstm_dims=lstm_sizes,
                     fc_size=fc_size, max_sent_len=max_sent_len,
                    )
model.build()
print("****model build finished***")

# # ==================train=================
# min_dev_loss = model_input.train(
#     model,
#     learning_rate,
#     train_x,
#     train_y,
#     val_x,
#     val_y,
#     max_epochs,
#     batch_size,
#     keep_prob,
#     l2reg,
#     show_step=show_step
# )
# logger.info(' ** The minimum dev_loss is {min_dev_loss}')

# ==================test =================
model_input.test(model, test_x, test_y, batch_size)
