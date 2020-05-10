# %%
import argparse
import json
import os
import threading
import time
from random import random

import numpy as np
import requests
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import numpy_encoder

#tf.random.set_random_seed(42)  # tensorflow seed fixing
# %%
'''
    - mnist dataset load & split
'''
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255

'''
#   reshape using CNN 
#train_images = train_images.reshape((-1, 28, 28, 1))
#test_images = test_images.reshape((-1, 28, 28, 1))
#input_shape = ((-1, 28, 28, 1))
'''

# %%
'''
    make by index list each number 
'''

train_index_list = [[], [], [], [], [], [], [], [], [], []]
test_index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    train_index_list[v].append(i)

for i, v in enumerate(test_labels):
    test_index_list[v].append(i)

# %%
'''
    build ann model
'''
def build_nn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

# %%
'''
    build cnn model 
'''
def build_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

# %%
def make_split_train_data_by_number(index_number, size=600):
    if index_number != -1 :
        random_index = np.random.randint(0, high=len(train_index_list[index_number]), size=size)

        s_train_image = []
        s_train_label = []
        for v in random_index:
            s_train_image.append(train_images[train_index_list[index_number][v]])
            s_train_label.append(train_labels[train_index_list[index_number][v]])
        return np.array(s_train_image), np.array(s_train_label)
    else:
        return train_images, train_labels

# %%
def make_split_train_data(size=600):
    random_index = np.random.randint(0, high=len(train_labels), size=size)
    s_train_image = []
    s_train_label = []
    for v in random_index:
        s_train_image.append(train_images[v])
        s_train_label.append(train_labels[v])
    return np.array(s_train_image), np.array(s_train_label)

# %%
def split_data(input_number= 0):
    temp_train, temp_labels = make_split_train_data_by_number(input_number, np.random.randint(1, 600))
    print("split data : input number : {} : size : {}".format(input_number, len(temp_labels)))
    return temp_train, temp_labels

# %%
'''
    request global_weight from server
'''
def request_global_weight():
    print("request_global_weight start")
    result = requests.get(ip_address)
    result_data = result.json()

    global_weight = None

    if result_data is not None:
        global_weight = []
        for i in range(len(result_data)):
            temp = np.array(result_data[i], dtype=np.float32)
            global_weight.append(temp)

    print("request_global_weight end")

    return global_weight

# %%
'''
    update local weight to server
'''
def update_local_weight(local_weight = []):
    print("update local weight start ")

    local_weight_to_json = json.dumps(local_weight, cls=numpy_encoder.NumpyEncoder)
    requests.put(ip_address, data=local_weight_to_json)

    print("update local weight end")

# %%
def train_local(global_weight = None):
    print("train local start")

    model = build_nn_model()

    global split_train_images
    global split_train_labels

    if global_weight is not None:
        global_weight = np.array(global_weight)
        model.set_weights(global_weight)

    model.fit(split_train_images, split_train_labels, epochs=5, batch_size=10, verbose=0)

    print("train local end")
    return model.get_weights()

# %%
def delay_compare_weight():
    print("current_round : {}, max_round : {}".format(current_round, max_round))
    if current_round < max_round:
        threading.Timer(delay_time, task).start()
    else:
        '''
        if input_number == 0:
            print_result()
        '''

# %%
def request_current_round():
    result = requests.get(request_round)
    result_data = result.json()

    return result_data

# %%
def validation(global_lound = 0, local_weight = []):
    print("validation start")
    if local_weight is not None:
        model = build_nn_model()
        model.set_weights(local_weight)

        result = model.predict(test_images)

        auroc_ovr = metrics.roc_auc_score(test_labels, result, multi_class='ovr')
        auroc_ovo = metrics.roc_auc_score(test_labels, result, multi_class='ovo')

        result = np.argmax(result, axis=1)
        cm = confusion_matrix(test_labels, result)
        acc = accuracy_score(test_labels, result)

        f1 = f1_score(test_labels, result, average=None)
        f2 = f1_score(test_labels, result, average='micro')
        f3 = f1_score(test_labels, result, average='macro')
        f4 = f1_score(test_labels, result, average='weighted')

        print("acc : {}".format(acc))
        print("auroc ovr : {}".format(auroc_ovr))
        print("auroc ovo : {}".format(auroc_ovo))
        print("f1 None : {}".format(f1))
        print("f2 micro : {}".format(f2))
        print("f3 macro : {}".format(f3))
        print("f4 weighted : {}".format(f4))

        print("cm : \n", cm)

        save_result(model, global_lound, global_acc=acc, f1_score=f2, auroc=auroc_ovo)

        print("validation end")


# %%
def save_result(model, global_rounds, global_acc, f1_score, auroc):
    test_name="FL4"
    create_directory("{}".format(test_name))
    create_directory("{}/model".format(test_name))

    if global_acc >= 0.8 :
        file_time = time.strftime("%Y%m%d-%H%M%S")
        model.save_weights("{}/model/{}-{}-{}-{}.h5".format(test_name, file_time, global_rounds, global_acc, f1_score))

    save_csv(test_name=test_name, round = global_rounds, acc = global_acc, f1_score=f1_score, auroc=auroc)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_csv(test_name = "", round = 0, acc = 0.0, f1_score = 0, auroc = 0):
    with open("{}/result.csv".format(test_name), "a+") as f:
        f.write("{}, {}, {}\n".format(round, acc, f1_score, auroc))

# %%

def task():
    print("--------------------")
    '''
        1. global weight request
        2. global weight & local weight compare
        3. start learning & validation processing            
        5. global weight update
        6. delay, next round
    '''
    global current_round

    global_round = request_current_round()
    print("global round : {}, local round :{}".format(global_round, current_round))

    if global_round == current_round:
        print("task train")
        # start next round
        global_weight = request_global_weight()
        local_weight = train_local(global_weight)

        # validation 0 clinet
        if input_number == 0 :
            validation(global_round, global_weight)

        update_local_weight(local_weight)
        delay_compare_weight()
        current_round += 1

    else:
        print("task retry")
        delay_compare_weight()

    print("end task")
    print("====================")

# %%
def print_result():
    print("====================")

# %%
from tensorflow.python.keras.callbacks import EarlyStopping

def single_train():

    early_stopping = EarlyStopping(patience=5)
    model = build_nn_model()

    model.fit(train_images, train_labels, epochs=1000, batch_size=32, verbose=1, validation_data=[test_images, test_labels], callbacks=[early_stopping])

    result = model.predict(test_images)
    result = np.argmax(result, axis=1)

    cm = confusion_matrix(test_labels, result)
    print("cm : ", cm)
    acc = accuracy_score(test_labels, result)
    print("acc : {}".format(acc))
    f1 = f1_score(test_labels, result, average=None)
    f2 = f1_score(test_labels, result, average='micro')
    f3 = f1_score(test_labels, result, average='macro')
    f4 = f1_score(test_labels, result, average='weighted')
    print("f1 : {}".format(f1))
    print("f2 : {}".format(f2))
    print("f3 : {}".format(f3))
    print("f4 : {}".format(f4))

# %%
if __name__ == "__main__":

    parameter = argparse.ArgumentParser()
    parameter.add_argument("--number", default=0)
    parameter.add_argument("--currentround", default=0)
    parameter.add_argument("--maxround", default=3000)
    args = parameter.parse_args()

    input_number = int(args.number)
    current_round = int(args.currentround)
    max_round = int(args.maxround)

    np.random.seed(42)
    np.random.seed(input_number)


    print("args : {}".format(input_number))

    global_round = 0
    delay_time = 5  # round check every 5 sec

    split_train_images, split_train_labels = split_data(input_number)

    base_url = "http://127.0.0.1:8000/"
    ip_address = "http://127.0.0.1:8000/weight"
    request_round = "http://127.0.0.1:8000/round"

    start_time = time.time()
    task()




