import json
from enum import Enum, auto

import numpy as np
import requests

from keras.models import load_model
from time import localtime, strftime
import tensorflow as tf

class Institution(Enum):
    SMC = auto()
    NNC = auto()
    ETC = auto()


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


class API():
    base_url = "http://FlServer.d6mm7kyzdp.ap-northeast-2.elasticbeanstalk.com"
    request_round_api = base_url + "/round"
    request_global_weight = base_url + "/weight"
    put_global_weight = base_url + "/weight"
    request_global_params = base_url + "/params"
    request_client_count = base_url + "/client_count"


class FL_Base:
    np.random.seed(19)    
    max_round = 10
    global_round = 0
    current_round = 0    
    delay_time = 10 # second
    local_epochs = 10
    local_batch_size = 100
    
    model = tf.keras.Sequential()
    
    def set_institution(self, institution):
        self.institution = institution

    def task(self):
        """task sequence

        1. global round request

        2. compare global round and local round

        3. if the round same, run validation

        4. start local training

        5. update local weight to aggregation server

        6. delay, do next round

        """
        pass

    def request_global_round(self):
        """
        request_global_round
        """
        print("> FL_Base: request_global_round")
        result = requests.get(API.request_round_api)
        self.global_round = result.json()
        print("------------------------------")
        print("> global round: {}, local round : {}".format(self.global_round, self.current_round))        
        print("------------------------------")
        return self.global_round

    def request_global_weight(self):
        """
        request_global_weight
        """
        print("> FL_Base: request_global_weight")
        self.global_weight = None
        
        result = requests.get(API.request_global_weight)
        result_data = result.json()

        if result_data is not None:
            self.global_weight = []
            for i in range(len(result_data)):
                temp = np.array(result_data[i], dtype=np.float32)
                self.global_weight.append(temp)

        return self.global_weight

    def local_training(self):
        """
        local_training
        """
        print("> FL_Base: local_training")

    def local_evaluate(self):
        """
        local_evaluate
        """
        print("> FL_Base: local_evaluate")

    def update_local_weight(self, local_weight=None):
        """
        update_local_weight
        """
        print("> FL_Base: update_local_weight")
        if local_weight is not None:
            local_weight_to_json = json.dumps(local_weight, cls=NumpyEncoder)
            result = requests.put(API.put_global_weight, data=local_weight_to_json)
        else:
            print("> FL_Base: update_local_weight: error: local_weight None")

        return result
    
    def save_model(self, file_path_name, ci_train, ci_test):
        print("> FL_Base: save_local_model")
        current_time = strftime("%y-%m-%d_%I:%M:%S", localtime())
        total_path = file_path_name + "_"+ str(ci_train) + "_" + str(ci_test) + "_" + current_time + ".h5"
        print("> FL_Base: save_model: " + total_path)
        self.model.save_weights(total_path)

    def delay_round(self):
        """
        delay_round

        if self.current_round < self.max_round:
        threading.Timer(self.delay_time, self.task).start()
        """
        print("> FL_Base: delay_round")

    '''
        check response object
    '''
    def request_global_params(self):
        print("> FL_Base: request_global_weight")
        result = requests.get(API.request_global_params)
        result_data = result.json()
        return result

    '''
        check response object
    '''
    def request_client_count(self):
        print("> FL_Base: request_client_count")
        result = requests.get(API.request_client_count)
        result_data = result.json()
        return result_data