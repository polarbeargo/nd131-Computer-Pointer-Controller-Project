source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
source venv/bin/activate
pip3 install -r requirements.txt
mkdir models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-retail-0004 -o models/ --precisions FP16
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 -o models/ --precisions FP16
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name facial-landmarks-35-adas-0002 -o models/ --precisions FP16
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 -o models/ --precisions FP16