using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace MLDemo
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly int inputWidth = 28, inputHeight = 28;
        private readonly string _modelPath = "../../../assets/mnist.onnx";
        private PredictionEngine<InputData, OutputData> _predictionEngine;

        public MainWindow()
        {
            InitializeComponent();
            InitializeModel();
        }

        private void InitializeModel()
        {
            MLContext context = new MLContext();
            var pipeline = context.Transforms.ApplyOnnxModel(_modelPath);

            var emptyData = new List<InputData>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var model = pipeline.Fit(data);

            _predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(model);
        }

        private void RecognizeDigit(object sender, RoutedEventArgs e)
        {
            var result = _predictionEngine.Predict(new InputData() { Image = ConvertInkCanvasToFloatArray() });
            numberLabel.Text = result.Result.ToList().IndexOf((float)result.Result.Max()).ToString();
        }

        private float[] ConvertInkCanvasToFloatArray()
        {
            // 获取InkCanvas的大小
            int width = (int)inkCanvas.ActualWidth;
            int height = (int)inkCanvas.ActualHeight;

            // 创建RenderTargetBitmap
            RenderTargetBitmap rtb = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Default);
            rtb.Render(inkCanvas);

            // 转换为8位灰度图
            FormatConvertedBitmap grayscaleBitmap = new FormatConvertedBitmap();
            grayscaleBitmap.BeginInit();
            grayscaleBitmap.Source = rtb;
            grayscaleBitmap.DestinationFormat = PixelFormats.Gray8; // 8-bit grayscale
            grayscaleBitmap.EndInit();

            // 将WPF的BitmapSource转换为System.Drawing.Bitmap
            Bitmap bitmap;
            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(grayscaleBitmap));
                enc.Save(outStream);
                bitmap = new Bitmap(outStream);
            }
            Bitmap bmp = ReSizeImage(bitmap, inputWidth, inputHeight);
            Bitmap graybmp = GetGaryImage(bmp);
            graybmp.Save("D:\\bitmap.bmp", ImageFormat.Bmp);
            return ConvertBitmapToFloatArray(graybmp);
        }

        private static Bitmap ReSizeImage(Image img, int width, int height)
        {
            Bitmap bitmap = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(bitmap);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.DrawImage(img, 0, 0, bitmap.Width, bitmap.Height);
            g.Dispose();
            return bitmap;
        }

        private static Bitmap GetGaryImage(Bitmap src)
        {
            float[][] colorMatrix = {
                new float[] {0.299f, 0.299f, 0.299f,     0,     0},
                new float[] {0.587f, 0.587f, 0.587f,     0,     0},
                new float[] {0.114f, 0.114f, 0.114f,     0,     0},
                new float[] {     0,      0,      0,     1,     0},
                new float[] {     0,      0,      0,     0,     1}
            };

            ImageAttributes ia = new ImageAttributes();
            ColorMatrix cm = new ColorMatrix(colorMatrix);
            ia.SetColorMatrix(cm, ColorMatrixFlag.Default, ColorAdjustType.Bitmap);

            Graphics g = Graphics.FromImage(src);
            g.DrawImage(
                src,
                new Rectangle(0, 0, src.Width, src.Height),
                0, 0,
                src.Width, src.Height,
                GraphicsUnit.Pixel,
                ia
            );
            g.Dispose();

            return src;
        }

        private float[] ConvertBitmapToFloatArray(Bitmap graybmp)
        {
            float[] graydata = new float[inputWidth * inputHeight];
            for (int i = 0; i < inputWidth; i += 1)
            {
                for (int j = 0; j < inputHeight; j += 1)
                {
                    System.Drawing.Color rescolor = graybmp.GetPixel(j, i);
                    graydata[(i * inputWidth) + j] = rescolor.R / 255.0f;
                }
            }
            return graydata;
        }

        private void ClearButtonClick(object sender, RoutedEventArgs e)
        {
            inkCanvas.Strokes.Clear();
            numberLabel.Text = "";
        }
    }

    public class InputData
    {
        [VectorType(1 * 28 * 28)]
        [ColumnName("Input3")]
        public float[] Image { get; set; }
    }

    public class OutputData
    {
        [ColumnName("Plus214_Output_0")]
        public float[] Result { get; set; }
    }
}
