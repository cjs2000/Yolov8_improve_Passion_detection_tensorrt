import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel , QFileDialog,QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,QThread, pyqtSignal
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from custom_buttons import *


from ctypes import *
import pyrealsense2 as rs
import numpy as np
import numpy.ctypeslib as npct
import time
import math
from PIL import Image

from yolov5 import YOLOv5

import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath

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

class ImageSelectorThread(QThread):
    file_selected = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.stopped = False

    def run(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                self.file_selected.emit(image_path)

    def stop(self):
        self.stopped = True


class CameraApp(QWidget):
    def __init__(self):
        super(CameraApp, self).__init__()
        self.setWindowTitle("UI")
        
        self.CameraConfig()
        self.UiConfig()
        
        #私有变量
        self.imagePath='' 
        self.fps=0.0
        self.LoadModel()
        self.TimerConfig()      



    def CameraConfig(self):
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        # 配置深度和颜色流  640,480
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
        align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
        align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
        # 启动RealSense管道
        self.pipeline.start(self.config)
        # 获取相机的内参
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


    def UiConfig(self):
        # 加载UI文件
        
        loadUi(r"UI/page.ui", self)
        self.setObjectName("wkWgt")
        self.setStyleSheet("#wkWgt{ background-color: #394a6e;}")

        self.picLabel.setAlignment(Qt.AlignCenter)

        self.imageButton.clicked.connect(self.toImageDetect)
        self.realButton.clicked.connect(self.toRealDetect)
        self.returnButton.clicked.connect(self.returnPage)
        self.backButton.clicked.connect(self.returnPage)
        self.selectButton.clicked.connect(self.selectPic)
        self.startButton.clicked.connect(self.startCap)
        self.stopButton.clicked.connect(self.stopCap)
        self.detectPicButton.clicked.connect(self.detectPic)
        self.takeButton.clicked.connect(self.openCap)
        self.cutButton.clicked.connect(self.cutPic)

        self.realButton.setObjectTheme(10)
        self.imageButton.setObjectTheme(10)
        self.realButton.setObjectTheme(10)
        self.returnButton.setObjectTheme(10)
        self.backButton.setObjectTheme(10)
        self.selectButton.setObjectTheme(10)
        self.startButton.setObjectTheme(10)
        self.stopButton.setObjectTheme(10)
        self.detectPicButton.setObjectTheme(10)
        self.takeButton.setObjectTheme(10)
        self.cutButton.setObjectTheme(10)


        self.cutButton.hide()
        self.picBox.addItem("yolov5n")
        self.picBox.addItem("yolov8n")
        self.picBox.addItem("yolov8n-improve")
        self.picBox.addItem("yolov8n-improve-P")
        self.picBox.addItem("yolov8n-improve-C")
        self.picBox.setCurrentIndex(3)
        self.picBox.currentIndexChanged.connect(self.on_picBox_changed)

        self.capBox.addItem("yolov5n")
        self.capBox.addItem("yolov8n")
        self.capBox.addItem("yolov8n-improve")
        self.capBox.addItem("yolov8n-improve-P")
        self.capBox.addItem("yolov8n-improve-C")
        self.capBox.setCurrentIndex(3)
        self.capBox.currentIndexChanged.connect(self.on_capBox_changed)
        
    
    def on_capBox_changed(self):
        self.stopCap()
        self.startCap()

    def on_picBox_changed(self):
        if self.imagePath == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No image has been selected!")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            id = self.picBox.currentIndex()
            if id == 0:
                self.yolov5n_pic()
            elif id == 1:
                self.yolov8n_pic()
            elif id == 2:
                self.yolov8n_improve_pic()
            elif id == 3:
                self.yolov8n_improve_engine_pic()
            elif id == 4:
                self.yolov8n_improve_C_pic()



    def cutPic(self):
        self.cutButton.hide()
        self.take_timer.stop()
        cv2.imwrite("cut.jpg", self.pic)
        self.imagePath="cut.jpg"


    def openCap(self):
        self.cutButton.show()
        self.take_timer.start(0)
        

    def takePic(self):
        capFrame = self.pipeline.wait_for_frames()
        color_frame = capFrame.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        self.pic=frame
        
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width,channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.picLabel.setPixmap(QPixmap.fromImage(qt_image))


    def TimerConfig(self):
        self.yolov5n_timer = QTimer(self)
        self.yolov5n_timer.timeout.connect(self.yolov5n_frame)
        self.yolov5n_timer.start(0)
        self.yolov5n_timer.stop()

        self.yolov8n_timer = QTimer(self)
        self.yolov8n_timer.timeout.connect(self.yolov8n_frame)
        self.yolov8n_timer.start(0)
        self.yolov8n_timer.stop()

        self.yolov8n_improve_timer = QTimer(self)
        self.yolov8n_improve_timer.timeout.connect(self.yolov8n_improve_frame)
        self.yolov8n_improve_timer.start(0)
        self.yolov8n_improve_timer.stop()

        self.yolov8n_improve_engine_timer = QTimer(self)
        self.yolov8n_improve_engine_timer.timeout.connect(self.yolov8n_improve_engine_frame)
        self.yolov8n_improve_engine_timer.start(0)
        self.yolov8n_improve_engine_timer.stop()

        self.yolov8n_improve_C_timer = QTimer(self)
        self.yolov8n_improve_C_timer.timeout.connect(self.yolov8n_improve_C_frame)
        self.yolov8n_improve_C_timer.start(0)
        self.yolov8n_improve_C_timer.stop()

        self.take_timer = QTimer(self)
        self.take_timer.timeout.connect(self.takePic)
        self.take_timer.start(0)
        self.take_timer.stop()



    def LoadModel(self):
        self.Yolov5n = YOLOv5("./v5n.pt",device='')  # 选择模型
        self.Yolov8n = YOLO("./v8n.pt")
        self.Yolov8n_improve = YOLO("./v8n-improve.pt")
        self.Yolov8n_improve_engine = YOLO("./v8n-improve.engine")
        self.det = Detector("/home/jetson/Desktop/tensorrt-yolov8/build/libyolov8_detect.so")

    def selectPic(self):
        '''
        # 打开文件对话框，选择图片文件
        # 创建并启动文件选择线程
        self.file_thread = ImageSelectorThread()
        self.file_thread.file_selected.connect(self.handle_file_selected)
        self.file_thread.start()
        '''
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            # 获取选中的文件路径
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                image_path = selected_files[0]
                # 显示选择的图片路径
                self.picLabel.setText(image_path)
                self.imagePath=image_path
                # 显示图片到 QLabel
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
                self.picLabel.setPixmap(scaled_pixmap)
        

    def handle_file_selected(self, image_path):
        if image_path:
            self.picLabel.setText(image_path)
            self.imagePath = image_path
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
            self.picLabel.setPixmap(scaled_pixmap)
             # 停止线程
        self.file_thread.stop()
        self.file_thread.wait()  # 等待线程结束



    def yolov5n_frame(self):
        capFrame = self.pipeline.wait_for_frames()
        color_frame = capFrame.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        depth_frame = capFrame.get_depth_frame()
        # 开始计时，用于计算帧率
        t1 = time.time()
        # 使用YOLOv5进行目标检测
        results = self.Yolov5n.predict(frame)
        count = 0
        # 在帧上绘制检测结果
        for *xyxy, conf, cls in results.xyxy[0]:
             count = count + 1
             label = f'{self.Yolov5n.model.names[int(cls)]} {conf:.2f}'
             center_x = (int(xyxy[0])+int(xyxy[2]))/2
             center_y = (int(xyxy[1])+int(xyxy[3]))/2
             # 计算颜色值，使颜色随着置信度变化而渐变
             Conf = int(conf * 255)
             color = (Conf, 255-Conf, 255-Conf*0.5)
             dist = depth_frame.get_distance(int(center_x), int(center_y))
             dist = round(dist, 2)
             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
             cv2.putText(frame, label + "dist:"+str(dist), (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)#cv2.FONT_HERSHEY_SIMPLEX
    
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % (self.fps) + " Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width,channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.capLabel.setPixmap(QPixmap.fromImage(qt_image))


    def yolov8n_frame(self):
        t1 = time.time()
        capFrame = self.pipeline.wait_for_frames()
        color_frame = capFrame.get_color_frame()
        depth_frame = capFrame.get_depth_frame()
        frame = np.asanyarray(color_frame.get_data())              #cap.read()
        results = self.Yolov8n.predict(source=frame) # 对当前帧进行目标检测并显示结果

        count = 0
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf)*255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            dist = depth_frame.get_distance(int(center_x), int(center_y))
            dist = round(dist, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255-Conf, 255-Conf*0.5), 2)
            cv2.putText(frame, "Passion:"+conf+" dist:"+str(dist), (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (Conf, 255-Conf, 255-Conf*0.5), 2 )
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2

        frame = cv2.putText(frame, "fps= %.2f" % (self.fps) + " Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width,channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.capLabel.setPixmap(QPixmap.fromImage(qt_image))

    def yolov8n_improve_frame(self):
        t1 = time.time()
        capFrame = self.pipeline.wait_for_frames()
        color_frame = capFrame.get_color_frame()
        depth_frame = capFrame.get_depth_frame()
        frame = np.asanyarray(color_frame.get_data())  # cap.read()
        results = self.Yolov8n_improve.predict(source=frame)  # 对当前帧进行目标检测并显示结果

        count = 0
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf) * 255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = depth_frame.get_distance(int(center_x), int(center_y))
            dist = round(dist, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255-Conf, 255-Conf*0.5), 2)
            cv2.putText(frame, "Passion:" + conf + " dist:" + str(dist), (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (Conf, 255-Conf, 255-Conf*0.5), 2)
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2

        frame = cv2.putText(frame, "fps= %.2f" % (self.fps) + " Count:" + str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.capLabel.setPixmap(QPixmap.fromImage(qt_image))


    def yolov8n_improve_engine_frame(self):
        t1 = time.time()
        capFrame = self.pipeline.wait_for_frames()
        color_frame = capFrame.get_color_frame()
        depth_frame = capFrame.get_depth_frame()
        frame = np.asanyarray(color_frame.get_data())  # cap.read()
        results = self.Yolov8n_improve_engine.predict(source=frame)  # 对当前帧进行目标检测并显示结果

        count = 0
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf) * 255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = depth_frame.get_distance(int(center_x), int(center_y))
            dist = round(dist, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255-Conf, 255-Conf*0.5), 2)
            cv2.putText(frame, "Passion:" + conf + " dist:" + str(dist), (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (Conf, 255-Conf, 255-Conf*0.5), 2)
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2

        frame = cv2.putText(frame, "fps= %.2f" % (self.fps) + " Count:" + str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.capLabel.setPixmap(QPixmap.fromImage(qt_image))


    def yolov8n_improve_C_frame(self):
        t1 = time.time()
        # 是否读取到了帧，读取到了则为True
        f1 = self.pipeline.wait_for_frames()
        color_frame = f1.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        depth_frame = f1.get_depth_frame()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image格式
        frame = Image.fromarray(np.uint8(frame))
        # 调整图片大小、颜色通道，使其适应YOLO推理的格式
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        # 推理图片
        result = self.det.predict(frame)
        count=0
        # 画框，标出识别的类别、距离、置信度等
        for temp in result:
            count=count+1
            store=temp[5]
            conf = str(temp[5])
            conf = conf[1:5]
            Conf = float(conf)*255
            x=int(temp[0])
            y=int(temp[1])
            w=int(temp[2])
            h=int(temp[3])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            dist = depth_frame.get_distance(center_x, center_y)
            dist = round(dist, 2)
            cv2.rectangle(frame,(int(temp[0]),int(temp[1])),(int(temp[0]+temp[2]),int(temp[1]+temp[3])), (Conf, 255-Conf, 255-Conf*0.5), 2)
            frame = cv2.putText(frame, "Passion:"+str(round(store,2))+" dist:"+str(dist),
                                (int(temp[0]),int(temp[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (Conf, 255-Conf, 255-Conf*0.5), 2) 
            
        # 计算帧率
        self.fps = (self.fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % (self.fps)+" Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.capLabel.setPixmap(QPixmap.fromImage(qt_image))

    def yolov8n_improve_C_pic(self):
        frame = cv2.imread(self.imagePath)
        frame = self.resize_keep_aspectratio(frame, (640, 480))
        cv2.imwrite("image1.jpg", frame)
        
        # 推理图片
        t1 = time.time()
        result = self.det.predict(frame)
        t = (time.time() - t1) * 1000
        t = round(t, 2)  # 保留两位小数
        t = str(t)  # 转换为字符串
        t = t + "ms"
        self.timeLabel.setText("RunTime: "+t)
        count=0
        # 画框，标出识别的类别、距离、置信度等
        for temp in result:
            count=count+1
            store=temp[5]
            conf = str(temp[5])
            conf = conf[1:5]
            Conf = float(conf)*255
            cv2.rectangle(frame,(int(temp[0]),int(temp[1])),(int(temp[0]+temp[2]),int(temp[1]+temp[3])), (Conf, 255-Conf, 255-Conf*0.5), 2)
            frame = cv2.putText(frame, "Passion:"+str(round(store,2)),
                                (int(temp[0]),int(temp[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (Conf, 255-Conf, 255-Conf*0.5), 2) 
        frame = cv2.putText(frame," Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
        self.picLabel.setPixmap(scaled_pixmap)
           
    


    def resize_keep_aspectratio(self,image_src, dst_size):
        src_h, src_w ,channel = image_src.shape
        fx = dst_size[0] / src_w
        fy = dst_size[1] / src_h
        dsize = (int(src_w * fx), int(src_h * fy))
        # 如果宽度大于高度，则缩小宽度
        if src_w > src_h:
            dsize = (dst_size[0], int(src_h * dst_size[0] / src_w))
        # 如果高度大于宽度，则缩小高度
        elif src_h > src_w:
            dsize = (int(src_w * dst_size[1] / src_h), dst_size[1])
        return cv2.resize(image_src, dsize, cv2.INTER_LINEAR)

    def detectPic(self):
        if self.imagePath == '':
            a=0
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No image has been selected!")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            id = self.picBox.currentIndex()
            if id == 0:
                self.yolov5n_pic()
            elif id == 1:
                self.yolov8n_pic()
            elif id == 2:
                self.yolov8n_improve_pic()
            elif id == 3:
                self.yolov8n_improve_engine_pic()
            elif id == 4:
                self.yolov8n_improve_C_pic()



    def yolov5n_pic(self):
        frame = cv2.imread(self.imagePath)
        frame = self.resize_keep_aspectratio(frame, (640,480))
        count = 0
        t1 = time.time()
        results = self.Yolov5n.predict(frame)
        t = (time.time() - t1) * 1000
        t = round(t, 2)  # 保留两位小数
        t = str(t)  # 转换为字符串
        t = t + "ms"
        self.timeLabel.setText("RunTime: "+t)
        for *xyxy, conf, cls in results.xyxy[0]:
             count = count + 1
             label = f'{self.Yolov5n.model.names[int(cls)]} {conf:.2f}'
             # 计算颜色值，使颜色随着置信度变化而渐变
             Conf = int(conf * 255)
             color = (Conf, 255-Conf, 255-Conf*0.5)
             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
             cv2.putText(frame, label , (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)#cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, " Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
        self.picLabel.setPixmap(scaled_pixmap)

    def yolov8n_pic(self):
        frame = cv2.imread(self.imagePath)
        frame = self.resize_keep_aspectratio(frame, (640,480))
        count = 0
        t1 = time.time()
        results = self.Yolov8n.predict(source=frame) # 对当前帧进行目标检测并显示结果
        t = (time.time() - t1) * 1000
        t = round(t, 2)  # 保留两位小数
        t = str(t)  # 转换为字符串
        t = t + "ms"
        self.timeLabel.setText("RunTime: "+t)
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf)*255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255-Conf, 255-Conf*0.5), 2)
            cv2.putText(frame, "Passion:"+conf, (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (Conf, 255-Conf, 255-Conf*0.5), 2)
        frame = cv2.putText(frame, " Count:"+str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
        self.picLabel.setPixmap(scaled_pixmap)

    def yolov8n_improve_pic(self):
        frame = cv2.imread(self.imagePath)
        frame = self.resize_keep_aspectratio(frame, (640, 480))
        count = 0
        t1 = time.time()
        results = self.Yolov8n_improve.predict(source=frame)  # 对当前帧进行目标检测并显示结果
        t = (time.time() - t1) * 1000
        t = round(t, 2)  # 保留两位小数
        t = str(t)  # 转换为字符串
        t = t + "ms"
        self.timeLabel.setText("RunTime: "+t)
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf) * 255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255-Conf, 255-Conf*0.5), 2)
            cv2.putText(frame, "Passion:" + conf, (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (Conf, 255-Conf, 255-Conf*0.5), 2)
        frame = cv2.putText(frame, " Count:" + str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
        self.picLabel.setPixmap(scaled_pixmap)

    def yolov8n_improve_engine_pic(self):
        frame = cv2.imread(self.imagePath)
        frame = self.resize_keep_aspectratio(frame, (640, 480))
        count = 0
        t1 = time.time()
        results = self.Yolov8n_improve_engine.predict(source=frame)  # 对当前帧进行目标检测并显示结果
        t = (time.time() - t1) * 1000
        t = round(t, 2)  # 保留两位小数
        t = str(t)  # 转换为字符串
        t = t + "ms"
        self.timeLabel.setText("RunTime: "+t)
        result = results[0]
        size = result.boxes.shape[0]
        for i in range(size):
            count = count + 1
            box = result.boxes[i]
            conf = str(box.conf.tolist())
            conf = conf[1:5]
            Conf = float(conf) * 255
            cords = box.xyxy[0].tolist()
            x1 = int(cords[0])
            y1 = int(cords[1])
            x2 = int(cords[2])
            y2 = int(cords[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (Conf, 255 - Conf, 255 - Conf * 0.5), 2)
            cv2.putText(frame, "Passion:" + conf, (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (Conf, 255 - Conf, 255 - Conf * 0.5), 2)
        frame = cv2.putText(frame, " Count:" + str(count), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 将OpenCV的图像格式转换为Qt图像格式
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.picLabel.width(), self.picLabel.height(), aspectRatioMode=1, transformMode=0)
        self.picLabel.setPixmap(scaled_pixmap)



    def startCap(self):
        id = self.capBox.currentIndex()
        if id == 0:
            self.yolov5n_timer.start(0)
        elif id == 1:
            self.yolov8n_timer.start(0)
        elif id == 2:
            self.yolov8n_improve_timer.start(0)
        elif id == 3:
            self.yolov8n_improve_engine_timer.start(0)
        elif id == 4:
            self.yolov8n_improve_C_timer.start(0)
        

        
    def stopCap(self):
        self.fps=0.0
        self.yolov5n_timer.stop()
        self.yolov8n_timer.stop()
        self.yolov8n_improve_timer.stop()
        self.yolov8n_improve_engine_timer.stop()
        self.yolov8n_improve_C_timer.stop()

    
    def toImageDetect(self):
        self.stackedWidget.setCurrentIndex(2)

    def toRealDetect(self):
        self.stackedWidget.setCurrentIndex(1)

    def returnPage(self):
        self.stopCap()
        self.take_timer.stop()
        self.stackedWidget.setCurrentIndex(0)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    camera_app = CameraApp()
   
    camera_app.show()
    sys.exit(app.exec_())
