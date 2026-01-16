namespace yolo26
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var FilePath = @"best.onnx";
            var yolo = new YoloSharp.Yolo(FilePath);
            Console.WriteLine("模型版本：" + yolo.Version);
        }
    }
}