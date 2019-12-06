# %%
import argparse
import json
import threading
import time

import numpy as np
import requests
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

import numpy_encoder

'''
    https://www.tensorflow.org/guide/gpu?hl=ko
    한대의 pc에서 실행하기 위해 gpu memory 사용에 제한을 둠 
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])

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
'''
    request global_weight from server
'''
def request_global_weight():
    print("request_global_weight start")
    result = requests.get(ip_address)
    result_data = result.json()

    global_weight = None

    #   Server에 global weight가 저장되어 있는 경우
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
def compare_global_local_weight():
    print("compare_global_local_weight start")

    status = False
    global global_weight
    global before_local_weight

    '''
        global weight와 before local weight 비교 함
        같은 경우 일정 시간 이후 다시 check_local_global_weight 수행
        다른 경우 train_local 수행
    '''
    # 초기 상태이면 pass
    if global_weight is None or len(before_local_weight) == 0:
        #print("before_local_weight None")
        return True

    global_np_weight = np.array(global_weight)
    local_np_weight = np.array(before_local_weight)
    compare_weight = global_np_weight - local_np_weight

    #print("comapre_wiehgt : ", compare_weight)

    if compare_weight != 0:
        #print("global weight == before local weight")
        print("")
    else:
        before_local_weight = global_weight
        #print("global weight != before local weight")
        status = True

    print("compare_global_local_weight end")
    return status

# %%
def train_validation_local(global_weight = None):
    print("train local start")

    local_start_time = time.time()

    td, tl = make_split_train_data_by_number(input_number, size=600)

    model = build_nn_model()

    if global_weight is not None:
        global_weight = np.array(global_weight)
        model.set_weights(global_weight)

    '''
        epochs = 2 >>> acc 상승 속도 느림 
        epochs = 5 >>>
    '''
    model.fit(td, tl, epochs=5, batch_size=10, verbose=0)

    validation_time_list.append(time.time() - local_start_time)

    print("train local end")
    return model.get_weights()

# %%
def delay_compare_weight():
    '''
            일정 시간 이후 task 재 호출
    '''
    if current_round is not max_round:
        threading.Timer(delay_time, task).start()
    else:
        print_result()

# %%
def request_current_round():
    result = requests.get(request_round)
    result_data = result.json()

    return result_data

# %%
def validation(local_weight = []):
    print("validation start")
    if local_weight is not None:
        model = build_nn_model()
        model.set_weights(local_weight)

        result = model.predict(test_images)
        result = np.argmax(result, axis=1)

        acc = accuracy_score(test_labels, result)
        print("acc : {}".format(acc))

        validation_acc_list.append(acc)
        #validation_loss_list.append(test_loss)
        print("validation end")

# %%

def task():
    print("--------------------")
    '''
        1. global weight request
        2. global weight & local weight 비교
        3. start learning & validation 진행             
        5. global weight update
        6. delay, next round
    '''
    global current_round

    global_round = request_current_round()
    print("global round : {}, local round :{}".format(global_round, current_round))

    if global_round == current_round:
        print("task train")
        # 다음 단계 진행
        global_weight = request_global_weight()
        local_weight = train_validation_local(global_weight)

        #   동일한 global weight를 사용하므로, 0번 client에서만 validation 진행 함
        if input_number == 0 :
            validation(global_weight)

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
    global start_time
    global validation_acc_list
    global validation_loss_list
    global validation_time_list

    print("====================")
    print("number : {}".format(input_number))
    #print("time : {}".format(time.time() - start_time))
    print("acc list", validation_acc_list)
    #print("loss list", validation_loss_list)
    #print("time list", validation_time_list)

# %%
def test():
    '''
        콘솔 디버깅용
    '''
    print("test start")
    global gw
    gw = request_global_weight()

    if gw is not None:
        model = build_nn_model()
        model.set_weights(gw)

        result = model.predict(test_images)
        result = np.argmax(result, axis=1)

        print("test end")

        cm = confusion_matrix(test_labels, result)
        print(cm)
        acc = accuracy_score(test_labels, result)
        print("acc : {}".format(acc))

# %%
if __name__ == "__main__":

    parameter = argparse.ArgumentParser()
    parameter.add_argument("number", default=0)
    args = parameter.parse_args()

    input_number = int(args.number)

    print("args : {}".format(input_number))

    max_round = 10
    global_round = 0
    delay_time = 5  #   5초마다 server&client 라운드 체크 진행 함
    current_round = 0

    validation_acc_list = []
    validation_loss_list = []
    validation_time_list = []

    base_url = "http://127.0.0.1:8080/"
    ip_address = "http://127.0.0.1:8000/weight"
    request_round = "http://127.0.0.1:8000/round"

    start_time = time.time()
    task()




