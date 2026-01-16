namespace YoloSharp
{
    /// <summary>
    /// 边界框类
    /// </summary>
    public class BoundingBox
    {
        /// <summary>
        /// 类别索引
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// 置信度
        /// </summary>
        public float Confidence { get; init; }

        /// <summary>
        /// 边界框左上角X坐标
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// 边界框左上角Y坐标
        /// </summary>
        public int Y { get; set; }

        /// <summary>
        /// 边界框宽度
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 边界框高度
        /// </summary>
        public int Height { get; set; }
    }
}