from argparse import ArgumentParser
import logging
from input_feeder import InputFeeder
import constants
import os
from face_detection import Face_Model
from facial_landmarks_detection import Model_Landmark
from gaze_estimation import Model_Gaze
from head_pose_estimation import Model_Pose
from mouse_controller import MouseController
import cv2
import imutils
import math


def imshow(windowname, frame, width=None):
    if width == None:
        width = 400

    frame = imutils.resize(frame, width=width)
    cv2.imshow(windowname, frame)


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--face", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-l", "--landmarks", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimation", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Provide the source of video frames." + constants.VIDEO + " " + constants.WEBCAM + " | " + constants.IP_CAMERA + " | " + constants.IMAGE)
    parser.add_argument("-debug", "--debug", required=False, type=str, nargs='+',
                        default=[],
                        help="To debug each model's output visually, type the model name with comma seperated after --debug")
    parser.add_argument("-ld", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="linker libraries if have any")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Provide the target device: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")

    return parser

    def main(args):

        feeder = None
        if args.input_type == constants.VIDEO or args.input_type == constants.IMAGE:
            extension = str(args.input).split('.')[1]
        if not extension.lower() in constants.ALLOWED_EXTENSIONS:
            print('Please provide supported extension.' +
                         str(constants.ALLOWED_EXTENSIONS))
            exit(1)

        if not os.path.isfile(args.input):
            print("Unable to find specified video/image file")
            exit(1)

            feeder = InputFeeder(args.input_type, args.input)
        elif args.input_type == constants.IP_CAMERA:
            if not str(args.input).startswith('http://'):
                print('Please provide ip of server with http://')
                exit(1)

            feeder = InputFeeder(args.input_type, args.input)
        elif args.input_type == constants.WEBCAM:
            feeder = InputFeeder(args.input_type)

        mc = MouseController("medium", "fast")

        feeder.load_data()

        face_model = Face_Model(args.face, args.device, args.cpu_extension)
        face_model.check_model()

        landmark_model = Model_Landmark(args.landmarks, args.device, args.cpu_extension)
        landmark_model.check_model()

        gaze_model = Model_Gaze(args.gazeestimation, args.device, args.cpu_extension)
        gaze_model.check_model()

        head_model = Model_Pose(
        args.headpose, args.device, args.cpu_extension)
        head_model.check_model()

        face_model.load_model()
        print("Face Detection Model Loaded...")
        landmark_model.load_model()
        print("Landmark Detection Model Loaded...")
        gaze_model.load_model()
        print("Gaze Estimation Model Loaded...")
        head_model.load_model()
        print("Head Pose Detection Model Loaded...")
        print('Loaded')
        try:

        except Exception as err:
        logger.error(err)

    cv2.destroyAllWindows()
    feeder.close()

    if __name__ == '__main__':

    args = build_argparser().parse_args()

    main(args)
