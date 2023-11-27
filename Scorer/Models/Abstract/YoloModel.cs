using System.Collections.Generic;
using Scorer.YoloParser;


namespace Scorer.Models.Abstract
{
    /// <summary>
    /// Model descriptor.
    /// </summary>
    public record YoloModel
    (
        int Width,
        int Height,
        int Depth,
        int Dimensions,

        float Confidence,
        float MulConfidence,
        float Overlap,

        int[] Strides,
        int[][][] Anchors,
        int[] Shapes,

        string[] Weights,
        List<YoloLabel> Labels,
        bool Ready
    );
}
