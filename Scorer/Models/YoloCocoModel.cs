using System.Collections.Generic;
using Scorer.Models.Abstract;
using Scorer.YoloParser;


namespace Scorer.Models
{
    /// <summary>
    /// ONNNX Model parameters and labels
    /// </summary>
    public class YoloCocoModel : YoloModel
    {
        public override int Width { get; } = 640;
        public override int Height { get; } = 640;
        public override int Depth { get; } = 3;

        /// <summary>
        /// Check onnx model Dimensions!
        /// </summary>
        public override int Dimensions { get; } = 17;
        public override float Confidence { get; } = 0.20f;
        public override float MulConfidence { get; } = 0.25f;
        public override float Overlap { get; } = 0.45f;
        public override string Weights { get; } = "Assets/weights/yolov5s.onnx";

        public YoloCocoModel()
        {
            Labels = new List<YoloLabel>()
            {
                new YoloLabel { Id = 1, Name = "stop" },
                new YoloLabel { Id = 2, Name = "leftturn" },
                new YoloLabel { Id = 3, Name = "rightturn" },
                new YoloLabel { Id = 4, Name = "rail" },
                new YoloLabel { Id = 5, Name = "35" },
                new YoloLabel { Id = 6, Name = "forward" },
                new YoloLabel { Id = 7, Name = "bike" },
                new YoloLabel { Id = 8, Name = "40" },
                new YoloLabel { Id = 9, Name = "ped" },
                new YoloLabel { Id = 10, Name = "xing" },
                new YoloLabel { Id = 11, Name = "keep" },
                new YoloLabel { Id = 12, Name = "clear" }
            };
        }
    }
}