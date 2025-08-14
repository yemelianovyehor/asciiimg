from sobel import sobel, grayscale
import numpy as np

term_size = (46, 135)


def detect_edge(image):


def edges(image, size=term_size):
    gray = grayscale(image)
    sobel_img = sobel(gray)
    img_dims = np.shape(sobel_img)
    chunk_dims = (img_dims[0]/size[0], img_dims[1], size[1])
    for i in range(1, size[0]):
        for j in range(1, size[1]):
            chunk = sobel_img[(i-1)*chunk_dims[0]:i*chunk_dims[0],
                              (j+1)*chunk_dims[1]:j*chunk_dims[1]]
