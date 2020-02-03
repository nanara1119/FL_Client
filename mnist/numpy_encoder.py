import json

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    '''
         통신에 필요한 serialize 과정 진행 (ndarray -> list)
     '''
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

