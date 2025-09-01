from matplotlib import pyplot as plt
import numpy as np
import asciiimg
import asciivid



class Image:
    def __init__(self, image):
        if isinstance(image, str):
            self.image = asciiimg.imread(image)
        elif isinstance(image, np.ndarray):
            self.image = image 
        
    def downscale(self, factor=8):
        self.image = asciiimg.downscale(self.image, factor)
        return self
    
    def grayscale(self, gamma=1.4):
        self.image = asciiimg.grayscale(self.image, gamma)
        return self
    
    def to_ascii(self, ascii_chars=asciiimg.ascii_chars, font_path="DejaVuSansMono.ttf", font_size=12,
                 text_color=255, bg_color=0):
        ascii_array = asciiimg.grayscale_to_ascii(self.image, ascii_chars)
        if (self.ascii_edges is not None):
            ascii_array = asciiimg.overlay_edges(ascii_array, self.ascii_edges)
        self.image = asciiimg.draw_ascii_image(ascii_array, font_path, font_size, text_color, bg_color)
        return self

    def sobel(self, magnitude_threshhold=0.3):
        self.edges = asciiimg.sobel(self.image, magnitude_threshhold)
        return self
    
    def deg2edges(self, cardinal_threshhold=10, edges = ("|","_","\\","/")):
        if self.edges is None:
            raise ValueError("Sobel edges not computed. Call sobel() before deg2edges().")
        self.ascii_edges = asciiimg.generate_edges(self.edges, cardinal_threshhold, edges)
        return self
    
    def draw(self, cmap=None):
        if cmap is None:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap=cmap)
        plt.axis("off")
        plt.show()
        return self

    def save(self, path, cmap=None):
        if cmap is None:
            plt.imsave(path, self.image) # type: ignore
        else:
            plt.imsave(path, self.image, cmap=cmap) # type: ignore
        return self


if __name__ == "__main__":
    img = Image("cat.jpg")
    img\
        .grayscale()\
        .downscale()\
        .sobel()\
        .deg2edges()\
        .to_ascii()\
        .draw()\
        .save("ascii_cat.png", cmap="gray")
        
