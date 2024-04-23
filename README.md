1. Use the code in the yolov8-train folder on your PC for model training.
   - Install Anaconda and create a virtual environment with Python version 3.9 and CUDA 12.1.
   - Within the virtual environment, execute `pip install ultralytics`.
   - Run `train.py` to start the training process.
   - Upon completion, the training will generate a model file named `best.pt`.

2. On the Jetson Nano platform:

   2.1. The YoloCap folder contains the code files for the detection system.
      - To run it, you will need to configure PyQt as well as software packages like torch, torchvision, yolov5, and ultralytics on Ubuntu.
      - First, install Anaconda and create a virtual environment with Python version 3.6.
      - Specifically, you will need to download the torch file that matches your development board's system from the Nvidia Jetson official website and execute `pip install "torch filename"` within the virtual environment.
      - It is not recommended to install the CPU version of the torch package, as the project will not be able to utilize the CUDA cores on the development board for detection, which will affect the inference speed.
      - After installing ultralytics, you will need to locate the ultralytics project folder under the Anaconda directory where you created the virtual environment. Replace all the files in it with the files from the yolov8-train folder of this project, otherwise, the improved model will not be loaded.
      - Once everything is ready, you will also need to configure files such as tensorrt to execute `python YoloCap.py`.

   2.2. For tensorrt:
      - To obtain the `.engine` file, follow these steps:
      - In the Python environment, navigate to the ultralytics folder in the terminal and execute `python toEngine.py` to get the Python version `.engine` file. Ensure the path to the `pt` file is set correctly. After obtaining the `.engine` file, place it in the YoloCap project folder. (Ensure that the tensorrt-python package is properly configured in the virtual environment.)
      - In the C++ environment, you will need to convert the improved training model to the ONNX format on the PC and then transfer it to the Jetson Orin Nano platform.
      - Execute the following command:
        ```
        /usr/src/tensorrt/bin/trtexec \
        --onnx=train/runs/detect/train/weights/best.onnx \
        --saveEngine=best.engine
        ```
      - After executing the above command, you will obtain an engine named `best.engine`.
      - Navigate to the tensorrt-yolov8 folder:
        ```
        cd ${root}/detect
        mkdir build
        cd build
        cmake ..
        make
        ```
      - You will then receive a file named `yolov8_detect.so`. Place both the `best.engine` and `yolov8_detect.so` into the YoloCap folder.
      - Under the YoloCap folder, enter the Anaconda virtual environment and execute `python YoloCap.py` to run the system.
