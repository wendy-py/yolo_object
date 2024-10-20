### Motivation
After troubleshooting a tutorial online, I re-coded to handle various PEP8 warnings. Then I added sky image without object and coded to handle crashes, so decided to share.

object features:
* build model from yolov3.cfg
* load pre-trained yolov3.weights
* perform object detection with image that contains objects
* perform object detection with image that has no object (test null)

### Install Guide
To test code, copy and run (assume linux with python3):
```
sudo apt-get install -y portaudio19-dev 
python3 -m venv test
cd test
source bin/activate
git clone https://github.com/wendy-py/yolo_object.git
cd yolo_object
wget https://pjreddie.com/media/files/yolov3.weights
pip install torch numpy pandas opencv-python matplotlib
python object.py
```
