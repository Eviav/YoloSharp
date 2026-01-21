namespace Winform
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        YoloSharp.Yolo? yolo;
        private void button1_Click(object sender, EventArgs e)
        {
            using (var dialog = new OpenFileDialog { Filter = "ONNX模型|*.onnx" })
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    yolo?.Dispose();
                    yolo = new YoloSharp.Yolo(dialog.FileName);
                    label1.Text = yolo.Description + " " + yolo.Version;
                    groupBox1.Enabled = true;
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (yolo == null) return;
            using (var dialog = new OpenFileDialog { Filter = "图片|*.jpg;*.jpeg;*.png" })
            {
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    var input = new YoloSharp.Input(yolo, Image.FromFile(dialog.FileName));

                    // 执行检测
                    var results = yolo.Detect(input);

                    // 处理检测结果
                    if (results != null)
                    {
                        foreach (var box in results)
                        {
                            MessageBox.Show($"类别: {yolo.Names[box.Index]}, 置信度: {box.Confidence}, 位置: ({box.X}, {box.Y}, {box.Width}, {box.Height})");
                        }
                    }
                }
            }
        }
    }
}
