from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ascii_chars = (" ", ".", ":", "c", "o", "P","0", "?", "@", "■")

def grayscale(image, gamma=1.4,
              consts={"r": 0.2126, "g": 0.7152, "b": 0.0722}):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale_image = np.matrix(r*consts["r"]**gamma + g * \
        consts["g"]**gamma + b*consts["b"]**gamma, dtype=np.float32)
    return grayscale_image

def downscale(image, factor):
    new_size = image.shape[0]//factor, image.shape[1]//factor
    new_image = np.empty_like(image)[0:new_size[0], 0:new_size[1]]
    if image.ndim == 2:
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                new_image[i, j] = image[i * factor : (i + 1) * factor,
                                        j * factor : (j + 1) * factor].mean()
    elif image.ndim == 3:
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                mean_col = image[i * factor : (i + 1) * factor,
                                 j * factor : (j + 1) * factor, :].mean(axis=(0,1))
                new_image[i, j] = mean_col
    return new_image

def grayscale_to_ascii(image, ascii_chars=ascii_chars):
    ascii_image = np.empty(image.shape, dtype=str)  # Create empty array for ASCII characters
    image = image * (len(ascii_chars)-1) // image.max()
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            ascii_image[i, j] =\
                ascii_chars[int(pixel_value)]  # Assign corresponding ASCII character
    return ascii_image

def draw_ascii_image(char_matrix, font_path="DejaVuSansMono.ttf", font_size=12, text_color=255, bg_color=0):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate character size
    bbox = font.getbbox('A')
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    img_width = char_matrix.shape[1] * char_width
    img_height = char_matrix.shape[0] * char_height

    img = Image.new('L', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    for y, row in enumerate(char_matrix):
        for x, char in enumerate(row):
            draw.text((x * char_width, y * char_height), str(char), font=font, fill=text_color)

    return img

def sobel(image, magnitude_threshhold:float=0):
    assert magnitude_threshhold >= 0 and magnitude_threshhold <= 1
    Gx = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    Gy = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
    
    gradient_x = np.zeros(image.shape)
    gradient_y = np.zeros(image.shape)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i-1:i+2, j-1:j+2]
            gradient_x[i, j] = np.sum(region * Gx)
            gradient_y[i, j] = np.sum(region * Gy)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = magnitude/magnitude.max()  # Normalize magnitude
    # filter = magnitude > magnitude_threshhold
    # G = np.zeros(image.shape)
    G = np.arctan2(gradient_x, gradient_y)  # Calculate angle
    G[magnitude < magnitude_threshhold] = None  # Filter out weak edges
    return np.rad2deg(G)

def generate_edges(degrees_array, cardinal_threshhold, edges = ("|","_","\\","/")):
    edges_array = np.empty(degrees_array.shape, dtype=str)
    for i in range(degrees_array.shape[0]):
        for j in range(degrees_array.shape[1]):
            angle = degrees_array[i, j]
            if np.isnan(angle):
                edges_array[i, j] = " "
            else:
                # Map angle to edge character
                deg = angle % 360
                if deg <= cardinal_threshhold:
                    edges_array[i, j] = edges[1] # _
                elif deg < 90 - cardinal_threshhold:
                    edges_array[i, j] = edges[2] # \
                elif deg <= 90 + cardinal_threshhold:
                    edges_array[i, j] = edges[0] # |
                elif deg < 180 - cardinal_threshhold:
                    edges_array[i, j] = edges[3] # /
                elif deg < 180 + cardinal_threshhold:
                    edges_array[i, j] = edges[1] # _
                elif deg < 270 - cardinal_threshhold:
                    edges_array[i, j] = edges[2] # \
                elif deg < 270 + cardinal_threshhold:
                    edges_array[i, j] = edges[0] # |
                elif deg < 360 - cardinal_threshhold:
                    edges_array[i, j] = edges[3] # /
                else:
                    edges_array[i, j] = edges[1] # _
    return edges_array

def overlay_edges(ascii_array, edges_array):
    np.copyto(edges_array, ascii_array, where=(edges_array == " "))
    return edges_array


def convert_img(input_file: str|np.ndarray, output_path="ascii_image.png",
                 downscale_factor=8, gamma=1.4,
                 magnitude_threshhold=0.3, cardinal_threshhold=10,
                 font_path="DejaVuSansMono.ttf", font_size=12,
                 text_color=255, bg_color=0, ascii_chars=ascii_chars):
    if isinstance(input_file, str):
        image = imread(input_file)
    elif isinstance(input_file, np.ndarray):
        image = input_file
    image = grayscale(image, gamma)
    image = downscale(image, downscale_factor)
    edges = generate_edges(sobel(image, magnitude_threshhold), cardinal_threshhold)
    ascii_image = grayscale_to_ascii(image, ascii_chars)

    img = overlay_edges(ascii_image, edges)
    img = draw_ascii_image(img, font_path, font_size, text_color, bg_color)
    if output_path is not None:
        img.save(output_path)
    return img
    

if __name__ == "__main__":
    # Example usage
    img = convert_img("cat.jpg", "ascii_image.png")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()