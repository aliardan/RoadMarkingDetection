using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Scorer.Models;
using Scorer.DataStructures;
using Scorer.YoloParser;

namespace RoadMarkingDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"..\\..\\..\\Assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var imagesFolder = Path.Combine(assetsPath, "input");
            var outputFolder = Path.Combine(assetsPath, "output");

            // Load Data
            IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
            
            Console.WriteLine("=========Identify the objects in the images=========");
            Console.WriteLine("====================================================");

            // iterate each image
            for (var i = 1; i <= images.Count(); i++)
            {
                var scorer = new YoloScorer<YoloCocoModel>(); using var stream = new FileStream($"Assets/input/{i}.jpg", FileMode.Open);
                var image = Image.FromStream(stream);

                List<YoloPrediction> predictions = scorer.Predict(image);
                using var graphics = Graphics.FromImage(image);

                Console.WriteLine($"=====Identify the objects in the image number {i}=====");
                Console.WriteLine("");

                // iterate each prediction to draw results
                foreach (var prediction in predictions)
                {
                    double score = Math.Round(prediction.Score, 2);

                    graphics.DrawRectangles(new Pen(prediction.Label.Color, 2),
                        new[] { prediction.Rectangle });

                    var (x, y) = (prediction.Rectangle.X - 2, prediction.Rectangle.Y - 21);

                    graphics.DrawString($"{prediction.Label.Name} ({score * 100}%)", new Font("TimesNewRoman", 14),
                        new SolidBrush(prediction.Label.Color), new PointF(x, y));
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
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}