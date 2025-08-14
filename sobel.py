from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


# consts in R,G,B
def grayscale(image, gamma=1.4,
              consts={"r": 0.2126, "g": 0.7152, "b": 0.0722}):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale_image = r*consts["r"]**gamma + g * \
        consts["g"]**gamma + b*consts["b"]**gamma
    return grayscale_image


'''
      _               _                   _                _
     |                 |                 |                  |
     | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
     | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
     |_               _|                 |_                _|
'''


def sobel(image):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    [row, col] = np.shape(image)
    filtered_img = np.zeros(shape=(row, col))

    for i in range(row-2):
        for j in range(col-2):
            gx = np.sum(np.multiply(Gx, image[i:i+3, j:j+3]))
            gy = np.sum(np.multiply(Gy, image[i:i+3, j:j+3]))
            filtered_img[i, j] = np.sqrt(gx**2+gy**2)
    return filtered_img


def main():
    path = "example.jpg"
    input_image = imread(path)
    [nx, ny, nz] = np.shape(input_image)
    gray_image = grayscale(input_image)
    filtered_image = sobel(gray_image)
    print(np.shape(gray_image))
    # filtered_image.tofile("edges_array.txt", sep="|")

    txt = ""
    for i in range(nx):
        for j in range(ny):
            if filtered_image[j, i] > 0.01:
                txt = txt.__add__(".")
            else:
                txt = txt.__add__(" ")
        txt = txt.__add__('\n')

    with open("text.txt", 'w') as f:
        f.write(txt)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax.imshow(filtered_image, cmap=plt.get_cmap("gray"))
    fig1.savefig("output.png")


main()
