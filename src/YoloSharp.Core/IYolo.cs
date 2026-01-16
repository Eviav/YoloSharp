using System;
using System.Collections.Generic;
using System.Drawing;

namespace YoloSharp
{
    /// <summary>
    /// YOLO目标检测接口
    /// </summary>
    public interface IYolo : IDisposable
    {
        /// <summary>
        /// 模型输入图像尺寸
        /// </summary>
        Size ImageSize { get; }

        /// <summary>
        /// 类别名称字典
        /// </summary>
        Dictionary<int, string> Names { get; }
    }
}