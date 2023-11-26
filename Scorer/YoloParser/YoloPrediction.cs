using SixLabors.ImageSharp;

namespace Scorer.YoloParser
{
    /// <summary>
    /// Object prediction.
    /// </summary>
    public record YoloPrediction(YoloLabel Label, float Score, RectangleF Rectangle);
}
