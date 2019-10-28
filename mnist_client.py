# %%
# import
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
    print("check local & global weight : {} /// {}".format(global_weight, before_local_weight))

    if global_weight == before_local_weight:
        print("aaaaaaaa")
    else:
        print("bbbbbbbb")

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
    print("update local weight start : ", type(weight), len(weight), len(weight[0]), type(weight[0]), len(weight[1]), type(weight[1]))

    local_weight_to_json = json.dumps(weight, cls=numpy_encoder.NumpyEncoder)
    result = requests.put(ip_address, data=local_weight_to_json)
    #print(result)
    print("update local weight end")

    '''
        weight update는 실패가 없다고 가정 ... 
    '''



# %%

def task():
    print("task start")

    model = build_nn_model()
    td, tl = make_split_train_data_by_number(0, size=1000)
    model.fit(td, tl, 10, 10, 0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("acc : {}, loss : {}".format(test_acc, test_loss))
    print("model weight len : ", len(model.get_weights()), type(model.get_weights()),len(model.get_weights()[0]), len(model.get_weights()[1]))
    for i in range(10) :
        update_local_weight(model.get_weights())
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

    #global_weight = get_global_weight()
    #print("get global weight : ", len(global_weight), len(global_weight[0]), len(global_weight[1]))

# %%
'''
    get global_weight from server
'''
def get_global_weight():
    result = requests.get(ip_address)
    global_weight = result.json()
    #   Server에 global weight가 저장되어 있지 않는 경우
    if len(global_weight) == 0:
        global_weight = None
    return global_weight





# %%
before_local_weight = []

if __name__ == "__main__":

    index = 0
    current_round = 0
    number_round = 100
    ip_address = "http://127.0.0.1:8000/weight"

    #before_local_weight = []
    #get_global_weight()
    task()





