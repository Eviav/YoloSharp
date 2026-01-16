using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;

namespace YoloSharp
{
    /// <summary>
    /// 工具辅助类
    /// </summary>
    public static class Helper
    {
        /// <summary>
        /// 解析尺寸字符串
        /// </summary>
        /// <param name="text">尺寸字符串，格式如 "[640, 640]"</param>
        /// <returns>解析后的Size对象</returns>
        public static Size ParseSize(string text)
        {
            text = text[1..^1]; // '[640, 640]' => '640, 640'
            var split = text.Split(", ");
            int y = int.Parse(split[0]), x = int.Parse(split[1]);
            return new Size(x, y);
        }

        /// <summary>
        /// 解析类别名称字符串
        /// </summary>
        /// <param name="names">类别名称字符串</param>
        /// <returns>类别名称字典</returns>
        public static Dictionary<int, string> ParseNames(this string names)
        {
            var nameList = names.TrimStart('{').TrimEnd('}').Split(", ");
            var list = new Dictionary<int, string>(nameList.Length);
            foreach (var it in nameList)
            {
                int index = it.IndexOf(":");
                if (int.TryParse(it.Substring(0, index), out int i))
                    list.Add(i, it.Substring(index + 2).Trim('\''));
            }
            return list;
        }

        /// <summary>
        /// 计算交并比（IoU）并应用非极大值抑制
        /// </summary>
        /// <param name="boxs">所有检测框列表</param>
        /// <param name="result">已筛选的检测框列表</param>
        /// <param name="box1">当前检测框</param>
        /// <param name="IoUThreshold">IoU阈值</param>
        /// <returns>是否保留当前检测框</returns>
        public static bool CalculateIoU(this List<RawBoundingBox> boxs, List<RawBoundingBox> result, RawBoundingBox box1, float IoUThreshold)
        {
            for (var j = 0; j < result.Count; j++)
            {
                var box2 = result[j];
                // 跳过不同类别的检测框
                if (box1.NameIndex != box2.NameIndex) continue;
                // 如果与已有检测框重叠度过高，则丢弃
                if (CalculateIoU(box1, box2) > IoUThreshold) return false;
            }
            return true;
        }

        /// <summary>
        /// 计算两个检测框的交并比（IoU）
        /// </summary>
        /// <param name="box1">第一个检测框</param>
        /// <param name="box2">第二个检测框</param>
        /// <returns>交并比数值</returns>
        public static float CalculateIoU(RawBoundingBox box1, RawBoundingBox box2)
        {
            var rect1 = box1.Bounds;
            var rect2 = box2.Bounds;

            var area1 = rect1.Width * rect1.Height;

            if (area1 <= 0f) return 0f; // 无效检测框

            var area2 = rect2.Width * rect2.Height;

            if (area2 <= 0f) return 0f; // 无效检测框

            // 计算交集区域
            var intersection = RectangleF.Intersect(rect1, rect2);
            var intersectionArea = intersection.Width * intersection.Height;

            // 计算IoU：交集面积 / (面积1 + 面积2 - 交集面积)
            return (float)intersectionArea / (area1 + area2 - intersectionArea);
        }

        /// <summary>
        /// 调整边界框到原始图像尺寸
        /// </summary>
        /// <param name="rectangle">模型输出的边界框</param>
        /// <param name="size">原始图像尺寸</param>
        /// <param name="model">模型输入尺寸</param>
        /// <returns>调整后的边界框</returns>
        public static Rectangle Adjust(RectangleF rectangle, Size size, Size model)
        {
            float xRatio = (float)size.Width / model.Width, yRatio = (float)size.Height / model.Height;
            float x = rectangle.X * xRatio, y = rectangle.Y * yRatio, w = rectangle.Width * xRatio, h = rectangle.Height * yRatio;
            return new Rectangle((int)x, (int)y, (int)w, (int)h);
        }

        /// <summary>
        /// 调整边界框到原始图像尺寸（重载）
        /// </summary>
        /// <param name="rectangle">模型输出的边界框</param>
        /// <param name="input_width">原始图像宽度</param>
        /// <param name="input_height">原始图像高度</param>
        /// <param name="model">模型输入尺寸</param>
        /// <returns>调整后的边界框</returns>
        public static Rectangle Adjust(RectangleF rectangle, int input_width, int input_height, Size model)
        {
            float xRatio = (float)input_width / model.Width, yRatio = (float)input_height / model.Height;
            float x = rectangle.X * xRatio, y = rectangle.Y * yRatio, w = rectangle.Width * xRatio, h = rectangle.Height * yRatio;
            return new Rectangle((int)x, (int)y, (int)w, (int)h);
        }

        /// <summary>
        /// 并行将RGB24数据转换为张量
        /// </summary>
        /// <param name="rgb24Data">RGB24数据</param>
        /// <param name="width">图像宽度</param>
        /// <param name="height">图像高度</param>
        /// <returns>转换后的张量（CHW格式，归一化到0-1）</returns>
        public static DenseTensor<float> ConvertRgb24ToTensorParallel(this byte[] rgb24Data, int width, int height)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });
            const float inv255 = 1.0f / 255.0f; // 归一化系数

            // 使用并行处理提高效率
            Parallel.For(0, height, y =>
            {
                int baseIndex = y * width * 3;
                for (int x = 0; x < width; x++)
                {
                    int index = baseIndex + x * 3;
                    // 将RGB值归一化到0-1，并转换为CHW格式
                    tensor[0, 0, y, x] = rgb24Data[index] * inv255; // R通道
                    tensor[0, 1, y, x] = rgb24Data[index + 1] * inv255; // G通道
                    tensor[0, 2, y, x] = rgb24Data[index + 2] * inv255; // B通道
                }
            });

            return tensor;
        }

        /// <summary>
        /// 快速将RGB24数据转换为张量（使用指针）
        /// </summary>
        /// <param name="rgb24Data">RGB24数据</param>
        /// <param name="size">图像尺寸</param>
        /// <returns>转换后的张量（CHW格式，归一化到0-1）</returns>
        public static unsafe DenseTensor<float> ConvertRgb24ToTensor(this byte[] rgb24Data, Size size)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, size.Height, size.Width });
            const float inv255 = 1.0f / 255.0f; // 归一化系数

            fixed (byte* rgbPtr = rgb24Data)
            fixed (float* tensorPtr = tensor.Buffer.Span)
            {
                byte* src = rgbPtr;
                // 分别指向R、G、B通道的起始位置
                float* rDest = tensorPtr;
                float* gDest = tensorPtr + size.Width * size.Height;
                float* bDest = tensorPtr + 2 * size.Width * size.Height;

                int pixelCount = size.Width * size.Height;
                for (int i = 0; i < pixelCount; i++)
                {
                    // 将RGB值归一化到0-1
                    *rDest++ = *src++ * inv255; // R通道
                    *gDest++ = *src++ * inv255; // G通道
                    *bDest++ = *src++ * inv255; // B通道
                }
            }
            return tensor;
        }

        /// <summary>
        /// 解析模型输出的边界框
        /// </summary>
        /// <param name="tensor">张量数据</param>
        /// <param name="boxStride">边界框步长</param>
        /// <param name="boxIndex">边界框索引</param>
        /// <param name="bounds">解析后的边界框</param>
        public static void ParseBox(this Span<float> tensor, int boxStride, int boxIndex, out RectangleF bounds)
        {
            // 从张量中读取边界框参数
            var x = tensor[0 + boxIndex];
            var y = tensor[1 * boxStride + boxIndex];
            var w = tensor[2 * boxStride + boxIndex];
            var h = tensor[3 * boxStride + boxIndex];

            // 转换为矩形框（x,y为中心点，转换为左上角坐标）
            bounds = new RectangleF(x - w / 2, y - h / 2, w, h);
        }
    }
}