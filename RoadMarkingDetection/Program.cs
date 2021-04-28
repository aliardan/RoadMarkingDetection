using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Scorer.Models;
using Microsoft.ML;
using Scorer.DataStructures;
using Scorer.YoloParser;

namespace RoadMarkingDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "Model", "yolov5s.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            // Initialize MLContext
            MLContext mlContext = new MLContext();

            // Load Data
            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
            IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

            var scorer = new YoloScorer<YoloCocoModel>();

            using var stream = new FileStream("assets/roadmark_0620.jpg", FileMode.Open);

            var image = Image.FromStream(stream);

            List<YoloPrediction> predictions = scorer.Predict(image);

            using var graphics = Graphics.FromImage(image);

            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            foreach (var prediction in predictions) // iterate each prediction to draw results
            {
                double score = Math.Round(prediction.Score, 2);

                graphics.DrawRectangles(new Pen(prediction.Label.Color, 2),
                    new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 2, prediction.Rectangle.Y - 21);

                graphics.DrawString($"{prediction.Label.Name} ({score * 100}%)", new Font("TimesNewRoman", 14),
                    new SolidBrush(prediction.Label.Color), new PointF(x, y));
                Console.WriteLine($"{prediction.Label.Name} and its Confidence score: {score * 100}%");
            }

            image.Save("assets/result.jpg");
            Console.WriteLine("");
            Console.WriteLine("=========End of Process..Hit any Key========");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}