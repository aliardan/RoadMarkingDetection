using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Scorer.Models;
using Scorer.DataStructures;
using Scorer.YoloParser;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RoadMarkingDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"Assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var imagesFolder = Path.Combine(assetsPath, "input");
            var outputFolder = Path.Combine(assetsPath, "output");

            FontCollection collection = new();
            FontFamily family = collection.Add("Assets/font/Arial.ttf");
            Font font = family.CreateFont(12, FontStyle.Regular);

            // Load Data
            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);

            Console.WriteLine("=========Identify the objects in the images=========");
            Console.WriteLine("====================================================");

            // iterate each image
            for (var i = 1; i <= images.Count(); i++)
            {
                var scorer = new YoloScorer<YoloRoadModel>("Assets/Weights/yolov5s.onnx");

                using var stream = new FileStream($"Assets/input/{i}.jpg", FileMode.Open);
                Image<Rgba32> image = Image.Load<Rgba32>(stream);

                var predictions = scorer.Predict(image);

                Console.WriteLine($"=====Identify the objects in the image number {i}=====");
                Console.WriteLine("");

                // iterate each prediction to draw results
                foreach (var prediction in predictions)
                {
                    double score = Math.Round(prediction.Score, 2);

                    var (x, y) = (prediction.Rectangle.Left - 3, prediction.Rectangle.Top - 23);

                    image.Mutate(x => x.DrawPolygon(Rgba32.ParseHex("#FFFF00"), 1,
                        new PointF(prediction.Rectangle.Left, prediction.Rectangle.Top),
                        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Top),
                        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Bottom),
                        new PointF(prediction.Rectangle.Left, prediction.Rectangle.Bottom)
                    ));

                    image.Mutate(a => a.DrawText($"{prediction.Label.Name} ({score * 100}%)", font, color: prediction.Label.Color, location: new PointF(x, y)));


                    Console.WriteLine($"{prediction.Label.Name} and its Confidence score: {score * 100}%");

                    image.Save($"{outputFolder}/result{i}.jpg");
                }

                Console.WriteLine("");

            }

            Console.WriteLine("=============End of Process..Hit any Key============");
        }

        /// <summary>
        /// Get Absolute Path
        /// </summary>
        /// <param name="relativePath"></param>
        /// <returns></returns>
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}