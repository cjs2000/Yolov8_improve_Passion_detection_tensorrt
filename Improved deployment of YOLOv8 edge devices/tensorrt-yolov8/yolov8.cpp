#include "chrono"
#include "include/yolov8.hpp"
#include "opencv2/opencv.hpp"




extern "C" {

const std::vector<std::string> CLASS_NAMES = {
	"Passion"};

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 255, 0 }, { 0, 0, 255 }
};

auto yolov8 = new YOLOv8("/home/jetson/Desktop/tensorrt-yolov8/best.engine");

void init()
{
    yolov8->make_pipe(true);
}


void Detect(int rows, int cols, unsigned char *src_data, float(*res_array)[6]){
    cv::Mat image = cv::Mat(rows, cols, CV_8UC3, src_data);
    cv::Size size = cv::Size{640, 640};
    std::vector<Object> objs;
    auto start = std::chrono::system_clock::now();
    objs.clear();
    
    yolov8->copy_from_Mat(image, size);
    
    yolov8->infer();
    yolov8->postprocess(objs);
    int j=0;
    for (const auto& obj : objs) {

        res_array[j][0] = obj.rect.x;
		res_array[j][1] = obj.rect.y;
		res_array[j][2] = obj.rect.width;
		res_array[j][3] = obj.rect.height;
		res_array[j][4] = obj.label;
		res_array[j][5] = obj.prob;
        j++;
    }

}

int detect()
{
    // 读取图片文件
    cv::Mat image = cv::imread("/home/jetson/Desktop/tensorrt-yolov8/image/a_13.jpg");
    cv::Mat res;

    // 检查图片是否正确读取
    if (image.empty())
    {
        std::cout << "无法读取图片文件" << std::endl;
        return -1;
    }
    
    auto yolov8 = new YOLOv8("/home/jetson/Desktop/tensorrt-yolov8/best.engine");
    
    yolov8->make_pipe(true);
    
    cv::Size size = cv::Size{640, 640};
    std::vector<Object> objs;
    auto start = std::chrono::system_clock::now();
    objs.clear();
    yolov8->copy_from_Mat(image, size);
    yolov8->infer();
    yolov8->postprocess(objs);
    for (const auto& obj : objs) {
        std::cout << "rect.x: " << obj.rect.x << std::endl;
        std::cout << "rect.y: " << obj.rect.y << std::endl;
        std::cout << "rect.width: " << obj.rect.width << std::endl;
        std::cout << "rect.height: " << obj.rect.height << std::endl;
        std::cout << "prob: " << obj.prob << std::endl;
        std::cout << "label: " << obj.label << std::endl;
        std::cout << std::endl;
    }


    yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);

    cv::imwrite("/home/jetson/Desktop/tensorrt-yolov8/image/a.jpg", res);
    return 0;

}
}




