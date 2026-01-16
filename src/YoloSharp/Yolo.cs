using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Drawing;

namespace YoloSharp
{
    /// <summary>
    /// YOLO目标检测核心类
    /// </summary>
    public class Yolo : IYolo
    {
        /// <summary>
        /// ONNX推理会话
        /// </summary>
        InferenceSession session;

        /// <summary>
        /// 初始化YOLO模型（CPU推理）
        /// </summary>
        /// <param name="path">模型文件路径</param>
        public Yolo(string path)
        {
            ModelPath = path;
            session = new InferenceSession(ModelPath);
            Names = ReadModel();
        }

        /// <summary>
        /// 初始化YOLO模型（GPU推理）
        /// </summary>
        /// <param name="path">模型文件路径</param>
        /// <param name="gpuid">GPU设备ID</param>
        public Yolo(string path, int gpuid)
        {
            ModelPath = path;
            session = new InferenceSession(ModelPath, SessionOptions.MakeSessionOptionWithCudaProvider(gpuid));
            Names = ReadModel();
        }

        #region 属性

        /// <summary>
        /// 模型文件路径
        /// </summary>
        public string ModelPath { get; private set; }

        /// <summary>
        /// 模型版本
        /// </summary>
        public string? Version { get; private set; }

        /// <summary>
        /// 是否为YOLO26模型
        /// </summary>
        public bool IsYOLO26 { get; private set; }

        /// <summary>
        /// 模型输入图像尺寸
        /// </summary>
        public Size ImageSize { get; private set; }

        /// <summary>
        /// 类别名称字典
        /// </summary>
        public Dictionary<int, string> Names { get; private set; }

        /// <summary>
        /// 置信度阈值
        /// </summary>
        public float Confidence { get; set; } = 0.3F;

        /// <summary>
        /// IoU（交并比）阈值
        /// </summary>
        public float IoUThreshold { get; set; } = 0.45F;

        #endregion

        #region 方法

        /// <summary>
        /// 读取模型元数据
        /// </summary>
        /// <returns>类别名称字典</returns>
        Dictionary<int, string> ReadModel()
        {
            var metadata = session.ModelMetadata.CustomMetadataMap;
            if (metadata.ContainsKey("description") && metadata["description"].Contains("YOLO26")) IsYOLO26 = true;
            if (metadata.ContainsKey("imgsz")) ImageSize = Helper.ParseSize(metadata["imgsz"]);
            if (metadata.ContainsKey("version")) Version = metadata["version"];
            return metadata.ContainsKey("names") ? metadata["names"].ParseNames() : new Dictionary<int, string>(0);
        }

        #endregion

        #region 算法推理

        /// <summary>
        /// 目标检测
        /// </summary>
        /// <param name="inputTensor">输入张量（CHW格式，已归一化到0-1）</param>
        /// <param name="input_width">原始输入图像宽度</param>
        /// <param name="input_height">原始输入图像高度</param>
        /// <returns>检测到的边界框列表</returns>
        public List<BoundingBox>? Detect(DenseTensor<float> inputTensor, int input_width, int input_height)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputNames[0], inputTensor) };
            using (var outputs = session.Run(inputs))
            {
                if (outputs[0].Value is DenseTensor<float> tensor)
                {
                    if (IsYOLO26)
                    {
                        // 检查维度: [1, 300, 6]，YOLO26模型输出格式
                        if (tensor.Dimensions.Length < 3 || tensor.Dimensions[2] != 6) return null;

                        int detectionsCount = tensor.Dimensions[1]; // 检测框数量
                        int featureSize = 6; // 每个检测框的特征数量：x1,y1,x2,y2,confidence,class
                        var tensorSpan = tensor.Buffer.Span;

                        var bs = new List<BoundingBox>();

                        for (int i = 0; i < detectionsCount; i++)
                        {
                            int offset = i * featureSize;
                            float score = tensorSpan[offset + 4]; // 置信度

                            if (score <= Confidence) continue; // 跳过置信度低的检测框

                            // 读取边界框坐标
                            float x1 = tensorSpan[offset + 0], y1 = tensorSpan[offset + 1], x2 = tensorSpan[offset + 2], y2 = tensorSpan[offset + 3];

                            // 计算边界框尺寸
                            RectangleF modelRect = new RectangleF(x1, y1, x2 - x1, y2 - y1);
                            // 调整边界框到原始图像尺寸
                            var rect = Helper.Adjust(modelRect, input_width, input_height, ImageSize);

                            bs.Add(new BoundingBox
                            {
                                Confidence = score,
                                Index = (int)tensorSpan[offset + 5], // 类别索引
                                X = rect.X,
                                Y = rect.Y,
                                Width = rect.Width,
                                Height = rect.Height
                            });
                        }

                        return bs;
                    }
                    else
                    {
                        // 处理其他YOLO模型输出格式
                        var boxStride = tensor.Strides[1];
                        var boxesCount = tensor.Dimensions[2];

                        var boxs = new List<RawBoundingBox>(boxesCount);
                        var tensorSpan = tensor.Buffer.Span;

                        for (var boxIndex = 0; boxIndex < boxesCount; boxIndex++)
                        {
                            for (var nameIndex = 0; nameIndex < Names.Count; nameIndex++)
                            {
                                // 读取置信度
                                var confidence = tensorSpan[(nameIndex + 4) * boxStride + boxIndex];

                                if (confidence <= Confidence) continue; // 跳过置信度低的检测框

                                // 解析边界框
                                tensorSpan.ParseBox(boxStride, boxIndex, out var bounds);

                                if (bounds.Width == 0 || bounds.Height == 0) continue; // 跳过无效边界框

                                boxs.Add(new RawBoundingBox
                                {
                                    Index = boxIndex,
                                    NameIndex = nameIndex, // 类别索引
                                    Confidence = confidence,
                                    Bounds = bounds
                                });
                            }
                        }

                        if (boxs.Count > 0)
                        {
                            // 按置信度降序排序
                            boxs.Sort((x, y) => y.CompareTo(x));
                            var result = new List<RawBoundingBox>(boxs.Count);

                            // 应用非极大值抑制（NMS）
                            for (var i = 0; i < boxs.Count; i++)
                            {
                                var box1 = boxs[i];
                                if (boxs.CalculateIoU(result, box1, IoUThreshold)) result.Add(box1);
                            }

                            // 转换为最终输出格式
                            var box = new List<BoundingBox>(result.Count);
                            foreach (var it in result)
                            {
                                var rect = Helper.Adjust(it.Bounds, input_width, input_height, ImageSize);
                                box.Add(new BoundingBox
                                {
                                    Confidence = it.Confidence,
                                    Index = it.NameIndex,
                                    X = rect.X,
                                    Y = rect.Y,
                                    Width = rect.Width,
                                    Height = rect.Height
                                });
                            }
                            return box;
                        }
                    }
                }
            }
            return null;
        }

        #endregion

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            session.Dispose();
        }
    }
}