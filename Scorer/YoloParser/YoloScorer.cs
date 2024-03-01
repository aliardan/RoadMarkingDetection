using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Scorer.Extensions;
using Scorer.Models.Abstract;
using System.Threading.Tasks;

namespace Scorer.YoloParser
{
    /// <summary>
    /// Object detector.
    /// </summary>
    public class YoloScorer<T> where T : YoloModel
    {
        private readonly T _model;
        private readonly InferenceSession _inferenceSession;

        /// <summary>
        /// Outputs value between 0 and 1.
        /// </summary>
        private float Sigmoid(float value)
        {
            return 1 / (1 + MathF.Exp(-value));
        }

        /// <summary>
        /// Converts xywh bbox format to xyxy.
        /// </summary>
        private static float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        /// <summary>
        /// Extracts pixels into tensor for net input.
        /// </summary>
        private static Tensor<float> ExtractPixels(Image<Rgba32> image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });

            Parallel.For(0, image.Height, y =>
            {
                Parallel.For(0, image.Width, x =>
                {
                    tensor[0, 0, y, x] = image[x, y].R / 255.0F; // r
                    tensor[0, 1, y, x] = image[x, y].G / 255.0F; // g
                    tensor[0, 2, y, x] = image[x, y].B / 255.0F; // b
                });
            });

            return tensor;
        }

        /// <summary>
        /// Runs inference session.
        /// Output tensors mapping
        /// </summary>
        private DenseTensor<float>[] Inference(Image<Rgba32> image)
        {
            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                image.Mutate(x => x.Resize(_model.Width, _model.Height)); // fit image size to specified input size
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(image))
            };

            var result = _inferenceSession.Run(inputs);

            DenseTensor<float>[] output =
            [
                result.First(x => x.Name == "397").Value as DenseTensor<float>, //output1//397//651
                result.First(x => x.Name == "417").Value as DenseTensor<float>, //output2//417//671
                result.First(x => x.Name == "437").Value as DenseTensor<float>  //output3//437//691
            ];

            return output;
        }

        /// <summary>
        /// Parses net output to predictions.
        /// </summary>
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            var result = new List<YoloPrediction>();

            var (xGain, yGain) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);
            var (xPadding, yPadding) = ((_model.Width - (image.Width * xGain)) / 2, (_model.Height - (image.Height * yGain)) / 2); // left, right pads

            for (int i = 0; i < output.Length; i++) // iterate outputs
            {
                int shapes = _model.Shapes[i]; // shapes per output

                for (int a = 0; a < _model.Anchors[i].Length; a++) // iterate anchors
                {
                    for (int y = 0; y < shapes; y++) // iterate rows
                    {
                        for (int x = 0; x < shapes; x++) // iterate columns
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            var objConfidence = buffer[4]; // extract object confidence

                            if (objConfidence < _model.Confidence) // check predicted object confidence
                                continue;

                            List<float> scores = buffer.Skip(5).Select(x => x * objConfidence).ToList();

                            float mulConfidence = scores.Max(); // find the best label

                            if (mulConfidence <= _model.MulConfidence) // check class obj_conf * cls_conf confidence
                                continue;

                            var rawX = ((buffer[0] * 2) - 0.5f + x) * _model.Strides[i]; // predicted bbox x (center)
                            var rawY = ((buffer[1] * 2) - 0.5f + y) * _model.Strides[i]; // predicted bbox y (center)

                            var rawW = MathF.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0]; // predicted bbox width
                            var rawH = MathF.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1]; // predicted bbox height

                            float[] xyxy = Xywh2xyxy([rawX, rawY, rawW, rawH]);

                            var xMin = Clamp((xyxy[0] - xPadding) / xGain, 0, image.Width - 0); // unpad, clip tlx
                            var yMin = Clamp((xyxy[1] - yPadding) / yGain, 0, image.Height - 0); // unpad, clip tly
                            var xMax = Clamp((xyxy[2] - xPadding) / xGain, 0, image.Width - 1); // unpad, clip brx
                            var yMax = Clamp((xyxy[3] - yPadding) / yGain, 0, image.Height - 1); // unpad, clip bry

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPrediction(label, mulConfidence, new(xMin, yMin, xMax - xMin, yMax - yMin));

                            result.Add(prediction);
                        }
                    }
                }
            }

            return [.. result];
        }

        /// <summary>
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items)
            {
                foreach (var current in result.ToList().Where(current => current != item)) // make a copy for each iteration
                {
                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    var intersection = RectangleF.Intersect(rect1, rect2);

                    var intArea = intersection.Area();
                    var unionArea = rect1.Area() + rect2.Area() - intArea;
                    var overlapRatio = intArea / unionArea;

                    if (overlapRatio >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Runs object detection.
        /// </summary>
        public List<YoloPrediction> Predict(Image<Rgba32> image)
        {
            return Suppress(ParseOutput(Inference(image.Clone()), image));
        }

        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights path and options.
        /// </summary>
        public YoloScorer(string weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights stream and options.
        /// </summary>
        public YoloScorer(Stream weights, SessionOptions opts = null) : this()
        {
            using var reader = new BinaryReader(weights);
            _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights bytes and options.
        /// </summary>
        public YoloScorer(byte[] weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
        }

        /// <summary>
        /// Disposes YoloScorer instance.
        /// </summary>
        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}