'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class Model_Landmark:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold = 0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.core = None
        self.network = None
        self.input = None
        self.output = None
        self.exec_network = None
        self.device = device
        self.threshold= threshold
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))
        self.inference_times = []
        self.processing_times = []

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        const = 10
        net_input = self.preprocess_input(image.copy())
        print('Hello')
        outputs = self.exec_network.infer({self.input:net_input})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        print('Hey dont stress')
        coords = coords[0]
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        l_xmin=coords[0]-const
        l_ymin=coords[1]-const
        l_xmax=coords[0]+const
        l_ymax=coords[1]+const
        
        r_xmin=coords[2]-const
        r_ymin=coords[3]-const
        r_xmax=coords[2]+const
        r_ymax=coords[3]+const
        cv2.rectangle(image,(l_xmin,l_ymin),(l_xmax,l_ymax),(255,0,0))
        cv2.rectangle(image,(r_xmin,r_ymin),(r_xmax,r_ymax),(255,0,0))
        cv2.imshow("Image",image)
        left_eye =  image[l_ymin:l_ymax, l_xmin:l_xmax]
        right_eye = image[r_ymin:r_ymax, r_xmin:r_xmax]
        eye_coords = [[l_xmin,l_ymin,l_xmax,l_ymax], [r_xmin,r_ymin,r_xmax,r_ymax]]
        return left_eye, right_eye, eye_coords

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = image.astype(np.float32)
        # net_input_shape = self.network.inputs[self.input].shape
        # p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        # print(np.shape(p_frame))
        # p_frame = p_frame.transpose((2, 0, 1))
        # print('work')
        # p_frame = p_frame.reshape(1, *p_frame.shape)
        (n, c, h, w) = self.network.inputs[self.input].shape
        frame = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        print((n,c,h,w))
        resized_frame = frame.transpose((2, 0, 1)).reshape((n, c, h, w))
        print('hi')
        return {self.input_name:resized_frame}
        
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        left_eye_x = outputs[0][0].tolist()[0][0]
        left_eye_y = outputs[0][1].tolist()[0][0]
        right_eye_x = outputs[0][2].tolist()[0][0]
        rght_eye_y = outputs[0][3].tolist()[0][0]
        return (left_eye_x, left_eye_y, right_eye_x, rght_eye_y)
