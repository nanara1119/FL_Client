# %%
# import
import argparse
import json
import threading

import numpy as np
import requests
import tensorflow as tf
import pandas as pd
import numpy_encoder
import base64

# %%
'''
    mnist load 및 (train, test), (image, label) 분리
'''
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255

#   cnn인 경우 reshape 필요
#train_images = train_images.reshape((-1, 28, 28, 1))
#test_images = test_images.reshape((-1, 28, 28, 1))
#input_shape = ((-1, 28, 28, 1))

# %%
'''
    각 숫자별로 묶은 index list 생성
'''

train_index_list = [[], [], [], [], [], [], [], [], [], []]
test_index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    train_index_list[v].append(i)

for i, v in enumerate(test_labels):
    test_index_list[v].append(i)

# %%
'''
    nn model build
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
    cnn model build
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
    random_index = np.random.randint(0, high=len(train_index_list[index_number]), size=size)

    s_train_image = []
    s_train_label = []
    for v in random_index:
        s_train_image.append(train_images[train_index_list[index_number][v]])
        s_train_label.append(train_labels[train_index_list[index_number][v]])
    return np.array(s_train_image), np.array(s_train_label)

# %%
'''
    Federated Learning 
'''

def run_federated():
    print("fl start")

    for i in range(number_round) :
        #model = build_cnn_model()
        model = build_nn_model()
        global_weight = get_global_weight()

        if global_weight is not None:
            model.set_weights(global_weight)


    print("fl end")


# %%
def check_local_global_weight():
    global_weight = get_global_weight()
    global before_local_weight


    if global_weight == before_local_weight:
        print("global weight == before local weight")
    else:
        print("global weight != before local weight")

    before_local_weight = global_weight

    '''
        global weight와 before local weight 비교 함
        같은 경우 일정 시간 이후 다시 check_local_global_weight 수행
        다른 경우 train_local 수행
    '''
    return False


# %%
def train_local(model):
    print("train local")

# %%

def update_local_weight(weight):
    print("update local weight start ")

    local_weight_to_json = json.dumps(weight, cls=numpy_encoder.NumpyEncoder)
    result = requests.put(ip_address, data=local_weight_to_json)

    print("update local weight end")

    '''
        weight update는 실패가 없다고 가정 ... 
    '''



# %%

def task():
    print("task start")
    model = build_nn_model()
    global input_number
    td, tl = make_split_train_data_by_number(input_number, size=1000)
    model.fit(td, tl, 10, 10, 0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("acc : {}, loss : {}".format(test_acc, test_loss))
    for i in range(10) :
        update_local_weight(model.get_weights())

    global lw
    lw = model.get_weights()
    '''
    if check_local_global_weight():
        train_local(None)
        update_local_weight(None)
    '''

    '''
        일정 시간 이후 task 재 호출
    '''
    '''
    if current_round is not number_round:
        threading.Timer(10, task).start()
    '''
    print("end task")

    global gw
    gw = get_global_weight()


# %%
'''
    get global_weight from server
'''
def get_global_weight():
    result = requests.get(ip_address)
    result_data = result.json()
    global_weight = []
    #   Server에 global weight가 저장되어 있지 않는 경우
    if len(result_data) == 0:
        global_weight = None

    for i in range(len(result_data)):
        temp = np.array(result_data[i], dtype=np.float32)
        global_weight.append(temp)

    return global_weight





# %%

before_local_weight = []
gw = []
lw = []
input_number = 0

if __name__ == "__main__":

    parameter = argparse.ArgumentParser()
    parameter.add_argument("number", default=0)
    args = parameter.parse_args()

    input_number = int(args.number)

    print("args : {}".format(input_number))

    index = 0
    current_round = 0
    number_round = 100
    ip_address = "http://127.0.0.1:8000/weight"


    #before_local_weight = []
    #get_global_weight()
    task()





