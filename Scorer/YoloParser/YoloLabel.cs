using SixLabors.ImageSharp;

namespace Scorer.YoloParser
{
    /// <summary>
    /// Label of detected object.
    /// </summary>
    public record YoloLabel(int Id, string Name, YoloLabelKind Kind, Color Color)
    {
        public YoloLabel(int id, string name) : this(id, name, YoloLabelKind.Generic, Color.Yellow) { }
    }
}
