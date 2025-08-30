import ffmpeg
import numpy as np
import shutil
import os

from asciiimg import convert_img

in_filename = 'example.mp4'

def convert_vid(input, output="example.mp4", framerate=25, downscale_factor=8, gamma=1.4,
                 magnitude_threshhold=0.3, cardinal_threshhold=10,
                 font_path="DejaVuSansMono.ttf", font_size=12,
                 text_color=255, bg_color=0):
    probe = ffmpeg.probe(input)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width']) # type: ignore
    height = int(video_stream['height']) # type: ignore

    out, _ = (
        ffmpeg
        .input(input)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )
    for frame in range(video.shape[0]):
        convert_img(video[frame], 
                    output_path=f"frames/frame_{frame:04d}.png",
                    downscale_factor=downscale_factor,
                    gamma=gamma,
                    magnitude_threshhold=magnitude_threshhold,
                    cardinal_threshhold=cardinal_threshhold,
                    font_path=font_path,
                    font_size=font_size,
                    text_color=text_color,
                    bg_color=bg_color)
        print(f"Processed frame {frame+1}/{video.shape[0]}")
        
    if not os.path.exists('frames') or not os.listdir('frames'):
        raise FileNotFoundError("No frames found in 'frames' directory")
    (
        ffmpeg
        .input('frames/frame_%04d.png', framerate=framerate)
        .output(output, pix_fmt='yuv420p', vcodec='libx264')
        .run()
    )