# %%
from __future__ import absolute_import
from __future__ import print_function

import csv
import json
import threading

import requests
import numpy as np
import argparse
import os
import imp
import time
import re

from keras.models import load_model
from scipy.stats import stats

from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import numpy_encoder, metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

from keras import backend as K
import pandas as pd
#from sklearn import metrics


address_request_round = "http://127.0.0.1:8000/round"
address_global_weight = "http://127.0.0.1:8000/weight"

global_round = 0
current_round = 0
max_round = 30
delay_time = 15

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../data/in-hospital-mortality/'))
                    #default="data/in-hospital-mortality/")

'''
    dim : hidden units
    depth : number of bi-LSTM's    
    dropout :     
    timestep : fixed timestep used in the dataset
    size_coef :

'''
parser.add_argument('--load_weight', type=str)
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print("args : " , args)

if args.small_part:
    args.save_every = 2**30

target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile_test.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'train_listfile_test.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

def build_model():
    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    suffix = ".bs{}{}{}.ts{}{}".format(args.batch_size,
                                       ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                       ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                       args.timestep,
                                       ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "")
    model.final_name = args.prefix + model.say_name() + suffix
    print("==> model.final_name:", model.final_name)

    # Compile the model
    print("==> compiling the model")
    optimizer_config = {'class_name': args.optimizer,
                        'config': {'lr': args.lr,
                                   'beta_1': args.beta_1}}

    # NOTE: one can use binary_crossentropy even for (B, T, C) shape.
    #       It will calculate binary_crossentropies for each class
    #       and then take the mean over axis=-1. Tre results is (B, T).
    if target_repl:
        loss = ['binary_crossentropy'] * 2
        loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    else:
        loss = 'binary_crossentropy'
        loss_weights = [1- args.target_repl_coef]

    model.compile(optimizer=optimizer_config,
                  loss=loss)
                  #loss_weights=loss_weights)
    model.summary()

    return model

model = build_model()

# Read data
print("start read train data")
train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part)
print("end read train data")

print("start read test data")
test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)
ret = utils.load_data(test_reader, discretizer, normalizer, args.small_part,
                      return_names=True)

data = ret["data"][0]
labels = ret["data"][1]
names = ret["names"]
print("end read test data")

if target_repl:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

'''---------------------------------------------------------------'''
def request_current_round():
    print("request_current_round")
    result = requests.get(address_request_round)
    result_data = result.json()
    print("request_current_round : ", result_data)
    return result_data

def request_global_weight():
    print("request_global_weight")
    result = requests.get(address_global_weight)
    result_data = result.json()

    global_weight = None

    if result_data is not None:
        global_weight = []
        for i in range(len(result_data)):
            temp = np.array(result_data[i])
            global_weight.append(temp)
        print("request_global_weight success : ", len(global_weight))
    return global_weight

def random_select_data(size = 600):
    random_index = np.random.randint(0, high=len(train_raw[0]), size = size)
    train_data = []
    train_label = []

    for v in random_index:
        train_data.append(train_raw[0][v])
        train_label.append(train_raw[1][v])

    return np.array(train_data), np.array(train_label)


def validation(global_weight=[]):
    if global_weight is not None:
        K.clear_session()
        model = build_model()
        model.set_weights(global_weight)

        print("==> validation start")
        predictions = model.predict(data, verbose=1)
        predictions = np.array(predictions)[:, 0]
        result = metrics.print_metrics_binary(labels, predictions)

        save_result(model, current_round, result)
        print("==> validation end")


def update_local_weight(local_weight = []):
    print("update_local_weight : ")
    local_weight_to_json = json.dumps(local_weight, cls = numpy_encoder.NumpyEncoder)
    requests.put(address_global_weight, local_weight_to_json)
    print("update_local_weight_to_json: ")

def delay_compare_weight():
    print("delay_compare_weight")
    if current_round is not max_round:
        threading.Timer(delay_time, task).start()

def save_result(model, current_round, result):
    test_name = "MIMIC"
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


# Prepare training
path = os.path.join(args.output_dir,
                    'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                          val_data=val_raw,
                                                          target_repl=(args.target_repl_coef > 0),
                                                          batch_size=args.batch_size,
                                                          verbose=args.verbose)
# make sure save directory exists
dirname = os.path.dirname(path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

keras_logs = os.path.join(args.output_dir, 'keras_logs')
if not os.path.exists(keras_logs):
    os.makedirs(keras_logs)
csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                       append=True, separator=';')



# %%
def task():
    print("========================================")
    if args.mode == 'train':
        '''
             python -um mimic3models.in_hospital_mortality.main 
             --network mimic3models/keras_models/lstm.py 
             --dim 16 
             --timestep 1.0 
             --depth 2 
             --dropout 0.3 
             --mode train 
             --batch_size 8 
             --output_dir mimic3models/in_hospital_mortality         

             python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality/lstm/baseline          
        '''
        K.clear_session()

        global current_round
        global val_raw
        global metrics_callback
        global saver
        global csv_logger

        print("==> training")

        global_round = request_current_round()
        if global_round == current_round:
            global_weight = request_global_weight()

            train_data = train_raw[0]
            train_label = train_raw[1]

            K.clear_session()

            if global_weight is not None:
                validation(global_weight)

            model = build_model()
            if global_weight is not None:
                model.set_weights(global_weight)

            print("==> train start")
            model.fit(train_data, train_label, epochs=2, batch_size=4, verbose=1)
            print("==> train end")
            update_local_weight(model.get_weights())
            delay_compare_weight()
            current_round += 1
        else:
            delay_compare_weight()


    elif args.mode == 'test':
        '''
            python -um mimic3models.in_hospital_mortality.main 
            --network mimic3models/keras_models/lstm.py 
            --dim 16 
            --timestep 1.0 
            --depth 2 
            --dropout 0.3 
            --mode test 
            --batch_size 8 
            --output_dir mimic3models/in_hospital_mortality 
            --load_state mimic3models\in_hospital_mortality\keras_states\k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch50.test0.2860894296052366.state

            python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models\in_hospital_mortality\keras_states\k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch50.test0.2860894296052366.state
            python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode test --batch_size 8 --output_dir mimic3models/in_hospital_mortality --load_state mimic3models\in_hospital_mortality\keras_states\k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch1.test0.3667576446537273
            
        '''
        # ensure that the code uses test_reader
        K.clear_session()
        model = build_model()
        model.load_weights("")  # saved model path&name

        print("==> validation start")

        validation(model.get_weights())

    else:
        raise ValueError("Wrong value for args.mode")


if __name__ == "__main__":
    task()
