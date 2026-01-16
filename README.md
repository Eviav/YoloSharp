# YoloSharp

YoloSharp 是一个基于 ONNX Runtime 的 YOLO 目标检测库，支持多种 YOLO 模型，包括 YOLO26，提供了简洁易用的 C# API。

> ⚠️ 目前仅支持目标检测，输入类型仅支持 Bitmap，正在开发中

> 暂无计划发布 Nuget 请使用源码，ML 的版本涉及到GPU环境限制，请按照自己的环境调整

## 功能特点

- 支持 CPU 和 GPU 推理
- 支持多种 YOLO 模型格式
- 提供简洁的 C# API
- 高性能的图像预处理
- 内置非极大值抑制（NMS）
- 支持边界框调整和归一化

## 项目结构

```
yolo-sharp/
├── src/
│   ├── YoloSharp/           # 核心检测类
│   ├── YoloSharp.Bitmap/    # Bitmap 图像处理
│   └── YoloSharp.Core/      # 核心接口和模型
│       └── Model/           # Class 类
├── test/                    # 测试项目
└── YoloSharp.slnx           # 解决方案文件
```

## 核心类说明

### Yolo 类
- **命名空间**: `YoloSharp`
- **功能**: 实现 YOLO 目标检测的核心类
- **构造函数**:
  - `Yolo(string path)`: 使用 CPU 初始化模型
  - `Yolo(string path, int gpuid)`: 使用 GPU 初始化模型
- **主要方法**:
  - `Detect(DenseTensor<float> inputTensor, int input_width, int input_height)`: 执行目标检测

### Input 类
- **命名空间**: `YoloSharp`
- **功能**: 处理 Bitmap 图像输入
- **主要方法**:
  - `Detection(IYolo yolo, Bitmap bmp)`: 将 Bitmap 转换为模型输入张量
  - `GetRGB(Bitmap bmp, out int stride)`: 从 Bitmap 获取 RGB 数据

### Helper 类
- **命名空间**: `YoloSharp`
- **功能**: 提供各种辅助方法
- **主要方法**:
  - `ParseSize(string text)`: 解析尺寸字符串
  - `ParseNames(string names)`: 解析类别名称
  - `CalculateIoU(RawBoundingBox box1, RawBoundingBox box2)`: 计算交并比
  - `ConvertRgb24ToTensor(byte[] rgb24Data, Size size)`: 转换 RGB 数据为张量

### 数据模型
- **BoundingBox**: 最终输出的边界框
- **RawBoundingBox**: 内部计算使用的原始边界框

## 依赖项

- Microsoft.ML.OnnxRuntime
- System.Drawing

## 使用示例

```csharp
using YoloSharp;
using System.Drawing;

// 初始化 YOLO 模型
var yolo = new Yolo("model.onnx");

// 加载图像
using var bitmap = new Bitmap("image.jpg");

// 创建输入处理器
var input = new Input();

// 执行检测
var tensor = input.Detection(yolo, bitmap);
var results = yolo.Detect(tensor, bitmap.Width, bitmap.Height);

// 处理检测结果
if (results != null)
{
    foreach (var box in results)
    {
        Console.WriteLine($"类别: {yolo.Names[box.Index]}, 置信度: {box.Confidence:.2f}, 位置: ({box.X}, {box.Y}, {box.Width}, {box.Height})");
    }
}
```

## 配置参数

- **Confidence**: 置信度阈值，默认 0.3
- **IoUThreshold**: IoU 阈值，默认 0.45

## 支持的模型

- YOLO26
- YOLO12
- 其他兼容 ONNX 格式的 YOLO 模型