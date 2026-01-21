using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloSharp
{
    /// <summary>
    /// YOLO输入接口
    /// </summary>
    public interface IInput
    {
        DenseTensor<float> DenseTensor { get; }
        int Width { get; }
        int Height { get; }
    }
}