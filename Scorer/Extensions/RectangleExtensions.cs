using System.Drawing;

namespace Scorer.Extensions
{
    public static class RectangleExtensions
    {
        /// <summary>
        /// Area of source
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static float Area(this RectangleF source)
        {
            return source.Width * source.Height;
        }
    }
}
