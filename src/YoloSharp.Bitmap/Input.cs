using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Drawing;
using System.Runtime.InteropServices;

namespace YoloSharp
{
    /// <summary>
    /// 图像输入处理类
    /// </summary>
    public class Input : IInput
    {
        /// <summary>
        /// 将图像转换为检测模型输入张量
        /// </summary>
        /// <param name="yolo">YOLO实例</param>
        /// <param name="image">输入图像</param>
        /// <remarks>支持Bitmap、Image等图像类型</remarks>
        public Input(IYolo yolo, Image image)
        {
            Width = image.Width;
            Height = image.Height;
            // 调整图像尺寸到模型输入大小
            using (var bmpCrop = new Bitmap(yolo.ImageSize.Width, yolo.ImageSize.Height))
            {
                using (var g = Graphics.FromImage(bmpCrop))
                {
                    g.DrawImage(image, 0, 0, bmpCrop.Width, bmpCrop.Height);
                }
                // 获取RGB数据并转换为张量（已处理归一化和CHW排列）
                DenseTensor = GetRGB(bmpCrop, out int stride).ConvertRgb24ToTensor(yolo.ImageSize);
            }
        }

        /// <summary>
        /// 从Bitmap获取RGB字节数据
        /// </summary>
        /// <param name="bmp">输入图像</param>
        /// <param name="stride">图像 stride（每行字节数）</param>
        /// <returns>RGB字节数组</returns>
        public static byte[] GetRGB(Bitmap bmp, out int stride)
        {
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            // 锁定图像内存并获取数据
            System.Drawing.Imaging.BitmapData bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            IntPtr ptr = bmpData.Scan0;
            int bytesLength = Math.Abs(bmpData.Stride) * bmp.Height;
            byte[] buffer = new byte[bytesLength];
            // 复制内存数据到字节数组
            Marshal.Copy(ptr, buffer, 0, bytesLength);
            // 解锁图像内存
            bmp.UnlockBits(bmpData);
            stride = bmpData.Stride;
            return buffer;
        }

        /// <summary>
        /// 模型输入张量，CHW格式
        /// </summary>
        public DenseTensor<float> DenseTensor { get; private set; }

        /// <summary>
        /// 原始图像宽度
        /// </summary>
        public int Width { get; private set; }

        /// <summary>
        /// 原始图像高度
        /// </summary>
        public int Height { get; private set; }
    }
}