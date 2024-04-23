
import ctypes
from ctypes import *
import cv2
import pyrealsense2 as rs
import numpy as np
import numpy.ctypeslib as npct
import time
import math
from PIL import Image

# 初始化RealSense管道
pipeline = rs.pipeline()
# 配置深度和颜色流
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动RealSense管道
pipeline.start(config)

# 获取相机的内参
intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()





class Detector():
    def __init__(self,dll_path):
        self.yolov8 = CDLL(dll_path)
        self.yolov8.init()
        self.yolov8.Detect.argtypes = [c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (50, 6), flags="C_CONTIGUOUS")]      

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50,6),dtype=np.float32)
        self.yolov8.Detect(c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array
        
det = Detector("/home/jetson/Desktop/tensorrt-yolov8/build/libyolov8_detect.so")



def function():
    fps = 0.0
    f1 = pipeline.wait_for_frames()
    color_frame = f1.get_color_frame()
    frame = np.asanyarray(color_frame.get_data())
    depth_frame = f1.get_depth_frame()
    depth_data = np.asanyarray(depth_frame.get_data())
    
    
    while frame.any:
        # 是否读取到了帧，读取到了则为True
        f1 = pipeline.wait_for_frames()
        color_frame = f1.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        depth_frame = f1.get_depth_frame()
        # 开始计时，用于计算帧率
        t1 = time.time()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image格式
        frame = Image.fromarray(np.uint8(frame))
        frame_shape = np.array(np.shape(frame)[0:2])
        # 调整图片大小、颜色通道，使其适应YOLO推理的格式
        
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # 推理图片
        result = det.predict(frame)
        
        count=0
        # 画框，标出识别的类别、距离、置信度等
        for temp in result:
            count=count+1
            bbox = [temp[0],temp[1],temp[2],temp[3]]  #xywh
            clas = int(temp[4])
            score = temp[5]
            middle_x = int(np.floor(temp[0]+temp[2]*0.5))
            middle_y = int(np.floor(temp[1]+temp[3]*0.5))
            x=int(temp[0])
            y=int(temp[1])
            w=int(temp[2])
            h=int(temp[3])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            dist = depth_frame.get_distance(center_x, center_y)
            dist = round(dist, 2)
            half_major_axis = int(w / 2)
            half_minor_axis = int(h / 2)
            frame = cv2.ellipse(frame, (center_x, center_y), (half_major_axis, half_minor_axis), 0, 0, 360, (0, 255, 0), 2)
            #cv2.rectangle(img,(int(temp[0]),int(temp[1])),(int(temp[0]+temp[2]),int(temp[1]+temp[3])), (0, 0, 225), 2)
            frame = cv2.putText(frame, "Passion"+" "+str(round(score,2))+"  dist:"+str(dist),
                              (int(temp[0]),int(temp[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 225), 2) 
        
        # 计算帧率
        fps = (fps + (1. / (time.time() - t1))) / 2
        #print("fps= %.2f" % (fps)) #控制台输出帧率
        frame = cv2.putText(frame, "fps= %.2f" % (fps)+" count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        #cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("frame", frame)

        # 若键盘按下q则退出播放
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

function()


