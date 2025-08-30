import argparse, os, shutil
from asciiimg import convert_img
from asciivid import convert_vid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASCII image converter')
    parser.add_argument('-i', type=str, help='Input image file path')
    parser.add_argument('-o', type=str, help='Output file path')
    parser.add_argument('--downscale-factor', default=8, type=int, help='Downscale factor for the output ASCII art')
    parser.add_argument('--gamma', default=1.4, type=float, help='Gamma correction value')
    parser.add_argument('--magnitude-threshhold', default=0.3, type=float, help='Magnitude threshold for edge detection')
    parser.add_argument('--cardinal-threshhold', default=10, type=float, help='Cardinal threshold for edge detection')
    parser.add_argument('--font-path', type=str, default="DejaVuSansMono.ttf", help='Path to the font file')
    parser.add_argument('--font-size', type=int, default=12, help='Font size for the output')
    parser.add_argument('--background', default=0, help='Background color')
    parser.add_argument('--text-color', default=255, help='Text color')
    parser.add_argument('-r', type=int, default=25, help='Framerate for video output (only for video input)')
    args = parser.parse_args()

    assert args.i is not None, "Input file path is required"
    assert args.o is not None, "Output file path is required"
    
    if (args.i.endswith('.mp4') or args.i.endswith('.gif')):
        if os.path.exists('frames'):
            shutil.rmtree('frames')
            os.makedirs('frames')
        convert_vid(args.i,
                    args.o,
                    framerate=args.r,
                    downscale_factor=args.downscale_factor,
                    gamma=args.gamma,
                    magnitude_threshhold=args.magnitude_threshhold,
                    cardinal_threshhold=args.cardinal_threshhold,
                    font_path=args.font_path,
                    font_size=args.font_size,
                    bg_color=args.background,
                    text_color=args.text_color)
    elif (args.i.endswith('.jpg') or args.i.endswith('.png')):
        convert_img(args.i,
                     output_path=args.o,
                     downscale_factor=args.downscale_factor,
                     gamma=args.gamma,
                     magnitude_threshhold=args.magnitude_threshhold,
                     cardinal_threshhold=args.cardinal_threshhold,
                     font_path=args.font_path,
                     font_size=args.font_size,
                     bg_color=args.background,
                     text_color=args.text_color)