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
        self.device = device
        self.threshold= threshold 
        
        model_bin = os.path.splitext(model_name)[0] + ".bin"
        try:
            self.network = IENetwork(model_name, model_bin)
        except Exception as e:
            print("Cannot initialize the network. Please enter correct model path. Error : %s", e)
        # self.network = self.core.read_network(model=str(model_name),
        #                                       weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.core = IECore()
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = []
        right_eye = []
        processed_image = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0,inputs={self.input_name: processed_image})
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            outputs= outputs[0]
            left_eye, right_eye, eye_coords = self.draw(outputs, image)
            
        return left_eye, right_eye, eye_coords

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" + str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def draw(self, outputs, image):
        const = 10
        outputs = outputs[0]
        h=image.shape[0]
        w=image.shape[1]
        outputs = outputs* np.array([w, h, w, h])
        outputs = outputs.astype(np.int32)
        l_xmin=outputs[0]-const
        l_ymin=outputs[1]-const
        l_xmax=outputs[0]+const
        l_ymax=outputs[1]+const
        
        r_xmin=outputs[2]-const
        r_ymin=outputs[3]-const
        r_xmax=outputs[2]+const
        r_ymax=outputs[3]+const
        cv2.rectangle(image,(l_xmin,l_ymin),(l_xmax,l_ymax),(255,0,0))
        cv2.rectangle(image,(r_xmin,r_ymin),(r_xmax,r_ymax),(255,0,0))
        cv2.imshow("Image",image)
        left_eye =  image[l_ymin:l_ymax, l_xmin:l_xmax]
        right_eye = image[r_ymin:r_ymax, r_xmin:r_xmax]
        eye_coords = [[l_xmin,l_ymin,l_xmax,l_ymax], [r_xmin,r_ymin,r_xmax,r_ymax]]
        
        return left_eye, right_eye, eye_coords

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        try:
            image = image.astype(np.float32)
            n,c,h,w = self.input_shape
            image = cv2.resize(image, (w,h))
            print(image.shape)
            image = image.transpose((2,0,1))
            print(image.shape)
            image = image.reshape(n,c,h,w)
            print('image')
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image
        
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
