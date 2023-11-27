using Scorer.Models.Abstract;


namespace Scorer.Models
{
    /// <summary>
    /// ONNNX Model parameters and labels
    /// </summary>
    public record YoloRoadModel() : YoloModel
    (
        640,
        640,
        3,

        /// <summary>
        /// Check onnx model Dimensions!
        /// </summary>
        17,
        0.20f,
        0.25f,
        0.45f,

        new[] { 8, 16, 32 },

        new[]
        {
            new[] { new[] { 010, 13 }, new[] { 016, 030 }, new[] { 033, 023 } },
            new[] { new[] { 030, 61 }, new[] { 062, 045 }, new[] { 059, 119 } },
            new[] { new[] { 116, 90 }, new[] { 156, 198 }, new[] { 373, 326 } }
        },

        new[] { 80, 40, 20 },

        new[] { "Assets/weights/yolov5s.onnx" },

        new()
        {
                new(1, "stop" ),
                new(2, "leftturn" ),
                new(3, "rightturn" ),
                new(4, "rail" ),
                new(5, "35" ),
                new(6, "forward" ),
                new(7, "bike" ),
                new(8, "40" ),
                new(9, "ped" ),
                new(10, "xing" ),
                new(11, "keep" ),
                new(12, "clear")
        },

    true
    );
}