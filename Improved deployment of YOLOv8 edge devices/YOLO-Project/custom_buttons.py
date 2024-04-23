from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QPushButton

class QQPushButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.a=1
        """
            self._animation = QtCore.QVariantAnimation()        实例化动画
            self._animation.setStartValue(0.00001)              启动的初始值
            self._animation.setEndValue(0.9999)                 结束值
            self._animation.valueChanged.connect(self._animate) 连接的函数
            self._animation.setDuration(500)                    动画延迟
            self._animation.loopCount()                         动画次数
            想要完成更好看的动画  还可以使用插值动画
        """
        self._animation = QtCore.QVariantAnimation()
        self._animation.setStartValue(0.00001)
        self._animation.setEndValue(0.9999)
        self._animation.valueChanged.connect(self._animate)
        self._animation.setDuration(500)

        """
        一些参数的设定
        """
        self.setObjectAnimatedOn = "hover"
        self.setIconAnimatedOn = None
        self.setObjectAnimate = "both"
        self.fallBackStyle = None
        self.defaultStyle = None
        self.clickPosition = None
        self.mousePosition = None
        self.applyShadowOn = None

        self.BianKuangCss = None
        self.Shaw = None
        """
        新加的边框样式
        """

    def setObjectTheme(self, theme):
        """
                下边是颜色主题
                按钮动态配色网站    https://gradientbuttons.colorion.co/
        """
        if str(theme) == "1":
            self.color1 = QtGui.QColor(9, 27, 27, 25)
            self.color2 = QtGui.QColor(85, 255, 255, 255)
        elif str(theme) == "2":
            self.color1 = QtGui.QColor(240, 53, 218)
            self.color2 = QtGui.QColor(61, 217, 245)
        elif str(theme) == "3":
            self.color1 = QtGui.QColor("#0e2515")
            self.color2 = QtGui.QColor("#4c5d4a")
        elif str(theme) == "4":
            self.color1 = QtGui.QColor("#FF16EB")
            self.color2 = QtGui.QColor("#100E19")
        elif str(theme) == "5":
            self.color1 = QtGui.QColor("#FF4200")
            self.color2 = QtGui.QColor("#100E19")
        elif str(theme) == "6":
            self.color1 = QtGui.QColor("#000046")
            self.color2 = QtGui.QColor("#1CB5E0")
        elif str(theme) == "7":
            self.color1 = QtGui.QColor("#EB5757")
            self.color2 = QtGui.QColor("#000000")
        elif str(theme) == "8":
            self.color1 = QtGui.QColor("#FF8235")
            self.color2 = QtGui.QColor("#30E8BF")
        elif str(theme) == "9":
            self.color1 = QtGui.QColor("#20002c")
            self.color2 = QtGui.QColor("#cbb4d4")
        elif str(theme) == "10":
            self.color1 = QtGui.QColor("#54ecd7")
            self.color2 = QtGui.QColor("#1D2671")
        elif str(theme) == "11":
            self.color1 = QtGui.QColor("#ee0979")
            self.color2 = QtGui.QColor("#ff6a00")
        elif str(theme) == "12":
            self.color1 = QtGui.QColor("#242424")
            self.color2 = QtGui.QColor("#FA0000")
        elif str(theme) == "13":
            self.color1 = QtGui.QColor("#25395f")
            self.color2 = QtGui.QColor("#55ffff")

        else:
            raise Exception("Unknown theme '" +str(theme)+ "'")
    def setObjectCustomTheme(self, color1, color2):
        self.color1 = QtGui.QColor(color1)
        self.color2 = QtGui.QColor(color2)
    def setObjectAnimation(self, animation):
        self.setObjectAnimate = str(animation)
    def setObjectAnimateOn(self, trigger):
        self.setObjectAnimatedOn = trigger
        if str(trigger) == "click":
            self._animation.setDuration(200) 
        else:
            self._animation.setDuration(500)
    """             """


    """
    重写事件 不用去鸟他 想研究的话建议备份
    """
    def enterEvent(self, event):
        pass
        self.mousePosition = "over"
        if self.setObjectAnimatedOn  == "hover" or self.setObjectAnimatedOn is None:
            self._animation.setDirection(QtCore.QAbstractAnimation.Forward)
            self._animation.start()
        #
        if self.setIconAnimatedOn == "hover":
            if hasattr(self, 'anim'):
                self.anim.start()
        if self.applyShadowOn == "hover":
            if self.animateShadow:
                self._shadowAnimation.setDirection(QtCore.QAbstractAnimation.Forward)
                self._shadowAnimation.start()

            else:
                self.setGraphicsEffect(self.shadow)
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.mousePosition = "out"
        if self.setObjectAnimatedOn == "hover" or self.setObjectAnimatedOn is None:
            self._animation.setDirection(QtCore.QAbstractAnimation.Backward)
            self._animation.start()
        super().leaveEvent(event)
    def mousePressEvent(self, event):
        self.clickPosition = "down"
        if self.setObjectAnimatedOn  == "click":
            self._animation.setDirection(QtCore.QAbstractAnimation.Forward)
            self._animation.start()
        if self.setIconAnimatedOn == "click":
            if hasattr(self, 'anim'):
                self.anim.start()
        if self.applyShadowOn == "click":
            if self.animateShadow:
                self._shadowAnimation.setDirection(QtCore.QAbstractAnimation.Forward)
                self._shadowAnimation.start()
            else:
                self.setGraphicsEffect(self.shadow)
        super().mousePressEvent(event)
    def mouseReleaseEvent(self, event):
        self.clickPosition = "up"
        if self.setObjectAnimatedOn  == "click":
            self._animation.setDirection(QtCore.QAbstractAnimation.Backward)
            self._animation.start()
        super().mouseReleaseEvent(event)
    """ """





    def _animate(self, value):
        """
        修改动态样式表的效果
        :param value: 接收的数值 为前面动画框架的数值
        :return:
        """
        if self.BianKuangCss == True:
            self.a = value
            ColorValue = int((value * 100) // 1)

            Color1 = 255
            Color2 = 255

            Qss = """   
              
                border:0px;
                border-bottom:8px solid qlineargradient(spread:pad, x1:"""+str(value - 0.01)+""", y1:1, x2:1, y2:1, 
                                stop:0.199005 rgb(0, """+str(Color1)+""", """+str(Color2)+"""), stop:0.21393 rgba(255, 255, 255, 0));
                border-left:8px solid qlineargradient(spread:pad, x1:0, y1:"""+str(1.1 - value)+""", x2:0, y2:0, 
                                stop:0.199005 rgb(0, """+str(Color1)+""", """+str(Color2)+"""), stop:0.21393 rgba(255, 255, 255, 0));
                border-right:8px solid qlineargradient(spread:pad, x1:1, y1:"""+str(value - 0.1)+""", x2:1, y2:1, 
                                stop:0.199005 rgb(0, """+str(Color1)+""", """+str(Color2)+"""), stop:0.21393 rgba(255, 255, 255, 0));
                border-top:8px solid qlineargradient(spread:pad, x1:"""+str(1.0001 - value)+""", y1:0, x2:0, y2:0, stop:0.199005 rgb(0, 255, 255), 
                                stop:0.21393 rgba(255, 255, 255, 0));
                color:rgb(0,"""+str(ColorValue + 150)+""","""+str(ColorValue + 150)+""");
                font: 75 24pt "Agency FB";
                
            """
            if self.Shaw == True:
                self.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(
                    blurRadius=ColorValue,
                    xOffset=0, yOffset=0,
                    color=QtGui.QColor(0, ColorValue + Color1 - 155, ColorValue + Color2 - 155)))
            #print(Qss)
            self.setStyleSheet(Qss)

            return
        color_stop = 1
        if self.defaultStyle is not None:
            qss = str(self.defaultStyle)
            #print(qss)
        else:
            qss = """
                border-style: solid;
                border-radius:5px;
                border-width: 2px;
                color: #d3dae3;
                padding: 5px;
            """
            #print(qss)
        grad = "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:{value} {color2}, stop: 1.0 {color1});".format(
            color1=self.color1.name(), color2=self.color2.name(), value=value
        )

        style = """      
            border-radius: 25px;   
            border-top-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.466, stop: """+str(value)+"""  """+str(self.color1.name())+""", stop: """+str(color_stop)+"""  """+str(self.color2.name())+""");
            border-bottom-color: qlineargradient(spread:pad, x1:1, y1:0.5, x2:0, y2:0.5, stop: """+str(value)+""" """+str(self.color1.name())+""", stop: """+str(color_stop)+"""  """+str(self.color2.name())+""");
            border-right-color: qlineargradient(spread:pad, x1:0.5, y1:0, x2:0.5, y2:1, stop:"""+str(value)+"""  """+str(self.color1.name())+""", stop: """+str(color_stop)+"""  """+str(self.color2.name())+""");
            border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop: """+str(value)+""" """+str(self.color1.name())+""", stop: """+str(color_stop)+"""  """+str(self.color2.name()) +""");

        """
        #print(style)
        if self.setObjectAnimate == "border":
            qss += style
        elif self.setObjectAnimate == "background":
            qss += grad
        else:
            qss += grad
            qss += style
        #print(qss)
        self.setStyleSheet(qss)

