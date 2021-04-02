import argparse
import collections
import datetime
import json
import os
import threading
import time

import architecture
import load
import numpy as np
import requests
import scipy.stats as sst
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tensorflow import keras
import sklearn.metrics as skm

import numpy_encoder

default_lr = 0.001
address_request_round = "http://127.0.0.1:8000/round"
address_global_weight = "http://127.0.0.1:8000/weight"
global_round = 0
current_round = 0
max_round = 100
delay_time = 15

train = load.load_dataset("data/train_2.json")
val = load.load_dataset("data/validation_2.json")
preproc = load.preproc(*train)

train_x, train_y = preproc.process(*train)
val_x, val_y = preproc.process(*val)

print("train size : {}, {}".format(len(train_x), len(train_y)))
print("val size : {}, {}".format(len(val_x), len(val_y)))


with open("data/validation_2.json", "rb") as fid:
    val_labels = [json.loads(l)['labels'] for l in fid]

counts = collections.Counter(preproc.class_to_int[l[0]] for l in val_labels)
counts = sorted(counts.most_common(), key=lambda x: x[0])
counts = list(zip(*counts))[1]

print("counts : " , counts)

smooth = 500
counts = np.array(counts)[None, None, :]
total = np.sum(counts) + counts.shape[1]
print("total : ", total)
prior = (counts + smooth) / float(total)    # ???
print("prior : ", prior)

def load_train_val_data(parser):
    print("load_train_val_data ... ")
    '''
    train = load.load_dataset("data/train.json")
    val = load.load_dataset("data/validation.json")
    preproc = load.preproc(*train)

    train_x, train_y = preproc.process(*train)
    val_x, val_y = preproc.process(*val)

    print("train size : {}, {}".format(len(train_x), len(train_y)))
    print("val size : {}, {}".format(len(val_x), len(val_y)))
    '''
    args = parser.parse_args()
    model = architecture.build_model()
    #print(model.summary())

    save_dir = make_save_dir("data/", "model")
    file_name = get_filename_for_saving(save_dir)
    check_pointer = keras.callbacks.ModelCheckpoint(
        filepath=file_name,
        save_best_only=False)
    stopping = keras.callbacks.EarlyStopping(patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=default_lr*0.001)

    model.fit(train_x, train_y,batch_size = int(args.batchsize), epochs = int(args.epochs),
              validation_data=(val_x, val_y))


def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{epoch:03d}-{val_loss:.3f}-{val_accuracy:.3f}-{loss:.3f}-{accuracy:.3f}.hdf5")

def make_save_dir(dirname, experiment_name):
    c_time = datetime.datetime.now()
    start_time = "{}{}{}{}{}".format(c_time.year, c_time.month, c_time.day, c_time.hour, c_time.minute)
    #start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

''' FL '''
def fl_task():
    print("fl task")
    global current_round

    global_round = request_current_round()
    if global_round == current_round:
        global_weight = request_global_weight()

        if global_weight is not None:
            validation(global_weight)

        model = architecture.build_model()

        if global_weight is not None:
            model.set_weights(global_weight)

        print("==> local training start")


        print("==> local training end")
        update_local_weight(model.get_weights())
        delay_compare_weight()
        current_round +=1

    delay_compare_weight()


def request_current_round():
    print("request current round")
    result = requests.get(address_request_round)
    result_data = result.json()
    print("reqeust_current_round : ", result_data)
    return result_data


def request_global_weight():
    print("request global weight")
    result = requests.get(address_global_weight)
    result_data = result.json()

    global_weight = None

    if result_data is not None:
        global_weight = []
        for i in range(len(result_data)):
            temp = np.array(result_data[i])
            global_weight.append(temp)
    return global_weight

def update_local_weight(local_weight = []):
    print("update local weight start")
    local_weight_to_json = json.dumps(local_weight, cls = numpy_encoder.NumpyEncoder)
    requests.put(address_global_weight, local_weight_to_json)
    print("update local weight end")

def validation(global_weight = []):
    if global_weight is not None:
        model = architecture.build_model()
        model.set_weights(global_weight)

        print("===> validation start")
        m_probs = model.predict(val_x)
        committee_labels = np.argmax(val_y, axis=2)
        committee_labels = committee_labels[:, 0]

        print("===================")
        temp = []
        preds = np.argmax(m_probs / prior, axis=2)
        for i, j in zip(preds, val_labels):
            t = sst.mode(i[:len(j) - 1])[0][0]
            temp.append(t)
            print(i[:len(j) - 1])

        preds = temp

        print("preds : \n", preds)

        report = skm.classification_report(committee_labels, preds, target_names=preproc.classes, digits=3)
        scores = skm.precision_recall_fscore_support(committee_labels, preds, average=None)
        print("report : \n", report)
        # print("scores : ", scores)

        cm = confusion_matrix(committee_labels, preds)
        print("confusion matrix : \n", cm)

        f1 = f1_score(committee_labels, preds, average='micro')
        print("f1_score : ", f1)

        # ***roc_auc_score - m_probs***

        m_probs = np.sum(m_probs, axis=1)
        m_probs = m_probs / 71  # one data set max size (element count) -> normalization

        # print(ground_truth.shape, m_probs.shape)

        ovo_auroc = roc_auc_score(committee_labels, m_probs, multi_class='ovo')
        ovr_auroc = roc_auc_score(committee_labels, m_probs, multi_class='ovr')

        print("ovr_auroc : ", ovr_auroc)
        print("ovo_auroc : ", ovo_auroc)

        result = {}

        save_result(model, current_round, result)
        
    print("===> validation end")


def delay_compare_weight():
    print("delay compare weight")
    if current_round is not max_round:
        threading.Timer(delay_time, fl_task).start()

def save_result(model, current_round, result):
    test_name = "ecg_base_3"
    create_directory("{}".format(test_name))
    create_directory("{}/model".format(test_name))

    file_time = time.strftime("%Y%m%d-%H%M%S")
    model.save_weights("{}/model/{}-{}-{:.4f}.h5".format(test_name, file_time, current_round, result["auroc"]))

    save_csv(test_name, current_round, result)

def save_csv(test_name, round = 0, result= {}):
    print("columns : ", result)
    with open("{}/result.csv".format(test_name), "a+") as f:
        f.write("{}, {}\n".format(round, result))

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def local_train():

    save_dir = make_save_dir("data/", "model")
    file_name = get_filename_for_saving(save_dir)

    check_pointer = keras.callbacks.ModelCheckpoint(
        filepath=file_name,
        save_best_only=False)
    stopping = keras.callbacks.EarlyStopping(patience=10)

    model = architecture.build_model()
    model.fit(train_x, train_y, batch_size=32,  epochs=30, validation_data=(val_x, val_y), callbacks=[check_pointer, stopping])
    validation(model.get_weights())


if __name__ == '__main__':
    print("start train")

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batchsize", default=32)

    start_time = time.time()
    load_train_val_data(parser)
    print("total time : ", (time.time() - start_time))
    '''

    local_train()


