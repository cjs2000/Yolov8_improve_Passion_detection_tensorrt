#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP

#include "fstream"
#include "common.hpp"
#include "NvInferPlugin.h"
using namespace det;

/**
 * @brief YOLOv8 TensorRT 推理类
 * 用于加载 .engine 模型文件、进行推理、结果后处理和可视化绘制。
 */
class YOLOv8
{
public:
    // 构造函数：加载 TensorRT engine
    explicit YOLOv8(const std::string& engine_file_path);
    // 析构函数：释放资源
    ~YOLOv8();

    // 初始化推理管线，分配显存与主机内存，可选择是否进行预热（warmup）
    void make_pipe(bool warmup = true);
    // 将 OpenCV 图像复制到 GPU 输入缓存（自动 letterbox 预处理）
    void copy_from_Mat(const cv::Mat& image);
    // 同上，但可手动指定目标输入尺寸
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    // 将输入图像进行 letterbox 缩放填充，保持纵横比不变
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    // 执行推理
    void infer();
    // 后处理推理结果，输出检测框信息
    void postprocess(std::vector<Object>& objs);

    // 绘制检测结果（目标框与标签）
    static void draw_objects(
        const cv::Mat& image,
        cv::Mat& res,
        const std::vector<Object>& objs,
        const std::vector<std::string>& CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS
    );

    // 绘制 FPS 信息到图像上
    static void draw_fps(
        const cv::Mat& image,
        cv::Mat& res,
        double infer_fps,
        int infer_rate
    );

    // 绑定数量统计
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;

    // 输入、输出绑定信息
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;

    // 主机与设备内存指针数组
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs;

    // 前处理参数（缩放比例、填充等）
    PreParam pparam;

private:
    // TensorRT 核心对象
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    // TensorRT 日志器
    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };
};

/**
 * @brief 构造函数：反序列化 .engine 文件并创建执行上下文
 */
YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    // 读取 .engine 文件
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    file.read(trtModelStream, size);
    file.close();

    // 初始化 TensorRT 插件系统
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 反序列化生成 engine
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    delete[] trtModelStream;
    assert(this->engine != nullptr);

    // 创建推理上下文
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 创建 CUDA 流
    cudaStreamCreate(&this->stream);

    // 获取绑定数量（输入 + 输出）
    this->num_bindings = this->engine->getNbBindings();

    // 遍历每个绑定，区分输入和输出
    for (int i = 0; i < this->num_bindings; ++i)
    {
        Binding binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string name = this->engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput)
        {
            this->num_inputs += 1;
            // 获取最大优化维度（Dynamic Shape 相关）
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // 设置输入维度
            this->context->setBindingDimensions(i, dims);
        }
        else
        {
            // 输出绑定
            dims = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

/**
 * @brief 析构函数：释放资源
 */
YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);

    for (auto& ptr : this->device_ptrs)
        CHECK(cudaFree(ptr));

    for (auto& ptr : this->host_ptrs)
        CHECK(cudaFreeHost(ptr));
}

/**
 * @brief 创建推理管线并预热
 */
void YOLOv8::make_pipe(bool warmup)
{
    // 分配输入显存
    for (auto& bindings : this->input_bindings)
    {
        void* d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    // 分配输出显存 + 主机页锁内存
    for (auto& bindings : this->output_bindings)
    {
        void* d_ptr, * h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // 预热推理 10 次
    if (warmup)
    {
        for (int i = 0; i < 10; i++)
        {
            for (auto& bindings : this->input_bindings)
            {
                size_t size = bindings.size * bindings.dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

/**
 * @brief 图像预处理：Letterbox 缩放 + padding
 */
void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    // 计算缩放比例 r
    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    // 缩放图像
    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
        cv::resize(image, tmp, cv::Size(padw, padh));
    else
        tmp = image.clone();

    // 计算填充区域
    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    // 边缘填充（灰色 114）
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // 转换为 NCHW 格式 blob
    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);

    // 保存预处理参数
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

/**
 * @brief 从 Mat 复制输入数据到 GPU
 */
void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat nchw;
    auto& in_binding = this->input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    // 更新输入维度
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    // 异步拷贝到 GPU
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

/**
 * @brief 从 Mat 复制输入数据（指定输入尺寸）
 */
void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

/**
 * @brief 执行推理（异步）
 */
void YOLOv8::infer()
{
    // 执行推理
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);

    // 从 GPU 异步拷贝结果到主机内存
    for (int i = 0; i < this->num_outputs; i++)
    {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }

    // 同步 CUDA 流
    cudaStreamSynchronize(this->stream);
}

/**
 * @brief 推理后处理：坐标映射回原图
 */
void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int* num_dets = static_cast<int*>(this->host_ptrs[0]);
    auto* boxes = static_cast<float*>(this->host_ptrs[1]);
    auto* scores = static_cast<float*>(this->host_ptrs[2]);
    int* labels = static_cast<int*>(this->host_ptrs[3]);

    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    for (int i = 0; i < num_dets[0]; i++)
    {
        float* ptr = boxes + i * 4;
        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        // 映射回原图坐标
        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        objs.push_back(obj);
    }
}

/**
 * @brief 绘制检测框与标签
 */
void YOLOv8::draw_objects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs, const std::vector<std::string>& CLASS_NAMES, const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs)
    {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;
        if (y > res.rows) y = res.rows;

        // 绘制标签背景
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

/**
 * @brief 绘制 FPS 信息框
 */
void YOLOv8::draw_fps(const cv::Mat& image, cv::Mat& res, double infer_fps, int infer_rate)
{
    int fpsCadreWidth = 200;
    int fpsCadreHeight = 70;
    int fpsCadreXPos = (res.cols - fpsCadreWidth) / 2;
    int fpsCadreYPos = 10;

    // 绘制黄色背景框
    rectangle(res, cv::Point(fpsCadreXPos, fpsCadreYPos), cv::Point(fpsCadreXPos + fpsCadreWidth, fpsCadreYPos + fpsCadreHeight), cv::Scalar(0, 255, 255), -1);

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 2;

    // 文本内容
    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(infer_fps));
    std::string frameTraitText = "1/" + std::to_string(infer_rate) + " frame processed";

    // 绘制 FPS
    cv::Size fpsTextSize = cv::getTextSize(fpsText, fontFace, fontScale, thickness, nullptr);
    int fpsTextX = (res.cols - fpsTextSize.width) / 2;
    int fpsTextY = fpsCadreYPos + (fpsCadreHeight - fpsTextSize.height) / 2;
    cv::putText(res, fpsText, cv::Point(fpsTextX, fpsTextY), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

    // 绘制帧率描述
    cv::Size frameTraitTextSize = cv::getTextSize(frameTraitText, fontFace, fontScale, thickness, nullptr);
    int frameTraitTextX = (res.cols - frameTraitTextSize.width) / 2;
    int frameTraitTextY = fpsTextY + fpsTextSize.height + 10;
    cv::putText(res, frameTraitText, cv::Point(frameTraitTextX, frameTraitTextY), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

#endif // JETSON_DETECT_YOLOV8_HPP
