using System;
using System.Drawing;
using Xunit;

namespace YoloSharp.Tests
{
    public class HelperTests
    {
        /// <summary>
        /// 测试尺寸字符串解析
        /// </summary>
        [Fact]
        public void ParseSizeTest()
        {
            // 测试正常情况
            var result = Helper.ParseSize("[640, 640]");
            Assert.Equal(new Size(640, 640), result);

            // 测试不同尺寸
            result = Helper.ParseSize("[1280, 720]");
            Assert.Equal(new Size(720, 1280), result);

            // 测试正方形尺寸
            result = Helper.ParseSize("[416, 416]");
            Assert.Equal(new Size(416, 416), result);
        }

        /// <summary>
        /// 测试类别名称解析
        /// </summary>
        [Fact]
        public void ParseNamesTest()
        {
            // 测试正常情况
            string namesStr = "{0: 'person', 1: 'bicycle', 2: 'car'}";
            var result = namesStr.ParseNames();

            Assert.Equal(3, result.Count);
            Assert.Equal("person", result[0]);
            Assert.Equal("bicycle", result[1]);
            Assert.Equal("car", result[2]);

            // 测试有效但无内容的字符串
            result = "{0: 'person'}".ParseNames();
            Assert.Single(result);
            Assert.Equal("person", result[0]);
        }

        /// <summary>
        /// 测试交并比计算
        /// </summary>
        [Fact]
        public void CalculateIoUTest()
        {
            // 创建测试边界框
            var box1 = new RawBoundingBox
            {
                Bounds = new RectangleF(0, 0, 100, 100)
            };

            var box2 = new RawBoundingBox
            {
                Bounds = new RectangleF(50, 50, 100, 100)
            };

            // 测试部分重叠
            var iou = Helper.CalculateIoU(box1, box2);
            Assert.Equal(0.14285715f, iou, 6); // 预期值：(50*50)/(100*100*2 - 50*50) = 2500/(20000-2500) = 2500/17500 = 1/7 ≈ 0.142857

            // 测试完全重叠
            var box3 = new RawBoundingBox
            {
                Bounds = new RectangleF(0, 0, 100, 100)
            };
            iou = Helper.CalculateIoU(box1, box3);
            Assert.Equal(1.0f, iou);

            // 测试完全不重叠
            var box4 = new RawBoundingBox
            {
                Bounds = new RectangleF(101, 101, 100, 100)
            };
            iou = Helper.CalculateIoU(box1, box4);
            Assert.Equal(0.0f, iou);

            // 测试无效边界框
            var box5 = new RawBoundingBox
            {
                Bounds = new RectangleF(0, 0, 0, 100)
            };
            iou = Helper.CalculateIoU(box1, box5);
            Assert.Equal(0.0f, iou);
        }

        /// <summary>
        /// 测试边界框调整
        /// </summary>
        [Fact]
        public void AdjustTest()
        {
            // 测试 Size 参数版本
            // 模型输出的边界框，坐标范围为 0 到 modelSize
            var rectangle = new RectangleF(0, 0, 320, 320); // 宽度和高度为模型尺寸的一半
            var originalSize = new Size(1280, 720);
            var modelSize = new Size(640, 640);

            var result = Helper.Adjust(rectangle, originalSize, modelSize);
            Assert.Equal(new Rectangle(0, 0, 640, 360), result);

            // 测试 width/height 参数版本
            result = Helper.Adjust(rectangle, 1280, 720, modelSize);
            Assert.Equal(new Rectangle(0, 0, 640, 360), result);

            // 测试不同位置和大小
            rectangle = new RectangleF(160, 160, 320, 320);
            result = Helper.Adjust(rectangle, 1280, 720, modelSize);
            Assert.Equal(new Rectangle(320, 180, 640, 360), result);
        }

        /// <summary>
        /// 测试边界框解析
        /// </summary>
        [Fact]
        public void ParseBoxTest()
        {
            // 创建测试张量数据 - 包含两个边界框，每个边界框4个参数
            float[] tensorData = new float[]
            {
                50, 50, 100, 100, // 第一个边界框：x, y, w, h
                150, 150, 200, 200  // 第二个边界框：x, y, w, h
            };

            var tensorSpan = tensorData.AsSpan();
            int boxStride = 1; // 每个边界框参数连续存储

            // 测试解析第一个边界框
            Helper.ParseBox(tensorSpan, boxStride, 0, out RectangleF bounds);
            Assert.Equal(new RectangleF(0, 0, 100, 100), bounds);

            // 测试解析第二个边界框 - 使用偏移后的索引
            Helper.ParseBox(tensorSpan.Slice(4), boxStride, 0, out bounds);
            Assert.Equal(new RectangleF(50, 50, 200, 200), bounds);
        }
    }
}