# Computer Pointer Controller Project  

[image1]: ./images/fdall.png    
[image2]: ./images/gei8.png
[image3]: ./images/gefp32.png 
[image4]: ./images/gefp16.png
[image5]: ./images/hpINT8.png
[image6]: ./images/hpFP16.png
[image7]: ./images/hpFP32.png
[image8]: ./images/landmarksINT8.png
[image9]: ./images/landmarksFP16.png
[image10]: ./images/landmarksFP32.png


In this project we controls the computer's mouse pointer with eye gaze.
We have used 4 pre-trained model that is provided by Open Model Zoo.
The project's main aim is to check usage of OpenVino ToolKit on different hardware
which includes the following:

- openvino inference API
- OpenVino DL WorkBench

## Project Set Up and Installation

1. Download [OpenVino ToolKit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) and install it locally.

2. Clone the repository from this URL: https://github.com/Polarbeargo/Computer-Pointer-Controller-Project.git
           
3. In Windows Command Prompt from working directory to install prerequisites or Execute <i>run.sh</i> from working directory to install prerequisites.

         cd Computer-Pointer-Controller
         pip install -r requirements.txt
         cd C:\Program Files (x86)\IntelSWTools\openvino\bin
         setupvars.bat
         or
         ./run.sh


## Demo  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NkXa3oqZm2Y/0.jpg)](https://youtu.be/NkXa3oqZm2Y)

Execute following command from the root directory of the project with your configuration.
    
    python starter/src/demo.py -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -lr models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -ge models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -d CPU -i starter/bin/demo.mp4 -flags ff fl fh fg

## Documentation

###### Command Line Arguments for Running the app

Argument|Type|Description
| ------------- | ------------- | -------------
-fd | Mandatory | Path to .xml file of Face Detection model.
-fl | Mandatory | Path to .xml file of Facial Landmark Detection model.
-fh| Mandatory | Path to .xml file of Head Pose Estimation model.
-fg| Mandatory | Path to .xml file of Gaze Estimation model.
-i| Mandatory | Path to video file or enter cam for webcam
-o| Mandatory | Specify path of output folder where we will store results.
-probs  | Optional | Specify confidence threshold which the value here in range(0, 1), default=0.6
-flags | Optional | ff for faceDetectionModel, fl for landmarkRegressionModel, fh for headPoseEstimationModel, fg for gazeEstimationMode
-d | Optional | Provide the target device: CPU / GPU / MYRIAD / FPGA

###### Directory Structure

- <b>bin</b> folder contains the media files
- <b>models</b> folder contains pre-trained models from Open Model Zoo
    - intel
        1. face-detection-adas-binary-0001
        2. gaze-estimation-adas-0002
        3. head-pose-estimation-adas-0001
        4. landmarks-regression-retail-0009
- <b>src</b> folder contains python files of the app
    
    + [demo.py](./starter/src/demo.py) : Main driver script to run the app
    + [face_detection.py](./starter/src/face_detection.py) : Face Detection related inference code
    + [facial_landmarks_detection.py](./starter/src/facial_landmarks_detection.py) : Landmark Detection related inference code
    + [gaze_estimation.py](./starter/src/gaze_estimation.py) : Gaze Estimation related inference code
    + [head_pose_estimation.py](./starter/src/head_pose_estimation.py) : Head Pose Estimation related inference code
    + [input_feeder.py](./starter/src/input_feeder.py) : input selection related code
    + [mouse_controller.py](./starter/src/mouse_controller.py) : Mouse Control related utilities.
    
- <b>README.md</b> File that you are reading right now.
- <b>requirements.txt</b> All the dependencies of the project listed here
- <b>run.sh</b> one shot execution script that covers all the prerequisites of the project.

## Benchmarks Results

Benchmark results on Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz Using OpenVINO DL Workbench the Benchmark performance were run on different quantization model(all models except Face Detection were available in three precisions: FP32, FP16, INT8):

* face-detection-adas-binary-0001:  
Face detection model is the bottleneck in this pipeline, 10 ms per frame means almost 93 FPS processing rate for video stream, which are acceptable in many use cases.
   
Precision: INT8, FP16, FP32  

![][image1]   

* gaze-estimation-adas-0002:  
In gaze estimation, we can see low performance in Precision FP32 with the higher latency and lower throughput compare to other models. The best precision model should be INT8 with lower latency and higher throughput in the gaze estimation usecase.    

Precision: INT8  
![][image2]  
Precision: FP16   
![][image4]  
Precision: FP32    
![][image3]  
* head-pose-estimation-adas-0001:  
In head pose estimation, we can see low performance in Precision FP16 with the higher latency and lower throughput compare to other models. The best precision model should be INT8 with lower latency and higher throughput in the head pose estimation usecase.  

Precision: INT8  
![][image5]  
Precision: FP16  
![][image6]  
Precision: FP32  
![][image7]  
* landmarks-regression-retail-0009:  

In landmarks regression, we can see low performance in Precision FP32 with the higher latency and lower throughput compare to other models. The best precision model should be INT8 with lower latency and higher throughput in the andmarks regression usecase.  

Precision: INT8  
![][image8]  
Precision: FP16  
![][image9]  
Precision: FP32  
![][image10]  
## Stand Out Suggestions  


* Async Inference:  

   * Using the start_async method will use the all cores of CPU improve performance with threading the ability to perform multiple inference at the same time compare to infer method. In synchrounous inference, the inference request need to be waiting until the other inference request executed therefore, it is more suitable to use async inference in this project.  

* Edge case:  
In this project demonstrate the ability to use human gaze to control computer mouse but there are some limitations:  
   * This project only work better when only one person has been detected in the frame. In the real condition, if use webcam, we should deal with detected multiple person. To deal with condision, we suggest to detect the main person only by combine Spreaker or Speech recognition to find main person who perform the computer pointer control role.
   * Some situations inference may break such as when Facial Landmark detection model returns empty image or the mouse move to the corner of the frame.  
   

* Use VTune Amplifier to measure hot spots in your application code.
