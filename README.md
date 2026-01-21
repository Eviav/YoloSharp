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
├── examples/                # 示例项目
│   └── Winform/             # Winform 示例
├── src/
│   ├── YoloSharp/           # 核心检测类
│   ├── YoloSharp.Bitmap/    # Bitmap 图像处理
│   └── YoloSharp.Core/      # 核心接口和模型
├── test/                    # 测试项目
│   ├── yolo12/              # YOLO12 测试
│   └── yolo26/              # YOLO26 测试
├── LICENSE                  # 许可证文件
├── README.md                # 项目说明文档
└── YoloSharp.slnx           # 解决方案文件
```

## 核心类说明

### Yolo 类
- **命名空间**: `YoloSharp`
- **功能**: 实现 YOLO 目标检测的核心类
- **构造函数**:
  - `Yolo(string path)`: 使用 CPU 初始化模型
  - `Yolo(string path, int gpuid)`: 使用 GPU 初始化模型
- **主要属性**:
  - `Confidence`: 置信度阈值，默认 0.3
  - `IoUThreshold`: IoU 阈值，默认 0.45
  - `Names`: 类别名称字典
  - `ImageSize`: 模型输入图像尺寸
- **主要方法**:
  - `Detect(IInput input)`: 执行目标检测，返回检测到的边界框列表

### Input 类
- **命名空间**: `YoloSharp`
- **功能**: 处理 Bitmap 图像输入，将图像转换为模型输入张量
- **构造函数**:
  - `Input(IYolo yolo, Image image)`: 将图像转换为模型输入张量
- **主要属性**:
  - `DenseTensor`: 模型输入张量（CHW格式）
  - `Width`: 原始图像宽度
  - `Height`: 原始图像高度
- **主要方法**:
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

### 基础使用示例

```csharp
using YoloSharp;
using System.Drawing;

// 初始化 YOLO 模型
var yolo = new Yolo("model.onnx");

// 加载图像
using var bitmap = new Bitmap("image.jpg");

// 执行检测
var input = new Input(yolo, bitmap);
var results = yolo.Detect(input);

// 处理检测结果
if (results != null)
{
    foreach (var box in results)
    {
        Console.WriteLine($"类别: {yolo.Names[box.Index]}, 置信度: {box.Confidence}, 位置: ({box.X}, {box.Y}, {box.Width}, {box.Height})");
    }
}
```

### 自定义检测参数

```csharp
using YoloSharp;
using System.Drawing;

// 初始化 YOLO 模型
var yolo = new Yolo("model.onnx");

// 设置自定义检测参数
yolo.Confidence = 0.5f; // 提高置信度阈值
yolo.IoUThreshold = 0.5f; // 调整 IoU 阈值

// 加载图像
using var bitmap = new Bitmap("image.jpg");

// 执行检测
var input = new Input(yolo, bitmap);
var results = yolo.Detect(input);

// 处理检测结果
if (results != null)
{
    Console.WriteLine($"检测到 {results.Count} 个目标:");
    foreach (var box in results)
    {
        string className = yolo.Names.TryGetValue(box.Index, out string name) ? name : "未知类别";
        Console.WriteLine($"- 类别: {className}, 置信度: {box.Confidence:F3}, 面积: {box.Width * box.Height:F1} 像素");
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

## 模型下载测试与转换

- 官方模型下载：`https://github.com/ultralytics/assets/releases`
- 转 ONNX：`https://docs.ultralytics.com/zh/tasks/detect/`

## 路线图

- 新增 OBB（语义分割）支持
- 新增 WPF 示例项目