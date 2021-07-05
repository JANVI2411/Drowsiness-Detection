from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import cv2
import time
from django.views.decorators.http import condition

from .sleep_detection_model.main import inference
from .sleep_detection_model.loader import initialize_parameters
from .sleep_detection_model.helper import gaze_tracking

def demo_webcam(request):
    return render(request,'sleep_detect/webcam.html')

def home(request):
    context = {}
    return render(request,'sleep_detect/test.html', context)

@gzip.gzip_page
def WebCam(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam),content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        raise

    context = {}
    return render(request,'sleep_detect/test.html', context)

class VideoCamera(object):
    """docstring forVideoCamera."""

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width=int(self.video.get(3))
        self.height=int(self.video.get(4))
        self.fps=int(self.video.get(5))
        self.params=initialize_parameters(self.width,self.height,self.fps)
        self.gaze=gaze_tracking()
        self.gaze.fps=self.fps

    def __del__(self):
        self.video.release()

    def get_frame(self,f_no):
        (self.grabbed, self.frame) = self.video.read()
        image = self.frame
        image=inference(image,f_no,self.gaze,self.params,self.width,self.height)
        _,jpeg = cv2.imencode('.jpg',image)
        return jpeg.tobytes()
        
def gen(camera):
    f_no=0
    while True:
        frame = camera.get_frame(f_no)
        yield(b'--frame\r\n'
              b'Content-Type: image /jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        f_no+=1
        
# To capture video class
# class VideoCamera(object):
#     """docstring forVideoCamera."""

#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         (self.grabbed, self.frame) = self.video.read()
#         threading.Thread(target=self.update,args=()).start()

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         image = self.frame
#         _,jpeg = cv2.imencode('.jpg',image)
#         return jpeg.tobytes()

#     def update(self):
#         while True:
#             (self.grabbed, self.frame) = self.video.read()
#             # if cv2.waitKey(1) == ord('q'):
#             #     print('break')
#             #     break
