using System.Drawing;

namespace Scorer.Extensions
{
    public static class RectangleExtensions
    {
        public static float Area(this RectangleF source)
        {
            return source.Width * source.Height;
        }
    }
}
