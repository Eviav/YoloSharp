using System.Drawing;

namespace YoloSharp
{
    /// <summary>
    /// 原始边界框类（用于内部计算）
    /// </summary>
    public class RawBoundingBox
    {
        /// <summary>
        /// 边界框索引
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// 类别索引
        /// </summary>
        public int NameIndex { get; init; }

        /// <summary>
        /// 置信度
        /// </summary>
        public float Confidence { get; init; }

        /// <summary>
        /// 边界框矩形
        /// </summary>
        public RectangleF Bounds { get; init; }

        /// <summary>
        /// 比较两个边界框的置信度
        /// </summary>
        /// <param name="other">另一个边界框</param>
        /// <returns>比较结果</returns>
        public int CompareTo(RawBoundingBox other) => Confidence.CompareTo(other.Confidence);
    }
}