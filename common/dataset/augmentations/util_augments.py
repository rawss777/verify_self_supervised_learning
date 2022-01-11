import random
from PIL import ImageFilter
from torch import nn
from torchvision import transforms


class RandomColorJitter(object):
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float, p: float):
        super(RandomColorJitter).__init__()
        self.transform = transforms.RandomApply([
            transforms.ColorJitter(brightness, contrast, saturation, hue)
            ], p=p)

    def __call__(self, img):
        return self.transform(img)


class GaussianBlur(object):
    def __init__(self, sigma: list):
        super(GaussianBlur).__init__()
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class RandomGaussianBlur(object):
    def __init__(self, sigma: list, p: float):
        super(RandomGaussianBlur).__init__()
        self.transform = transforms.RandomApply([
            GaussianBlur(sigma)
            ], p=p)

    def __call__(self, img):
        return self.transform(img)

