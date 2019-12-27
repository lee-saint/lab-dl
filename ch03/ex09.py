"""
PIL 패키지와 numpy 패키지를 이용하면,
이미지 파일(jpg, png, bmp, ...)의 픽셀 정보를 numpy.ndarray 형식으로 변환하거나
numpy.ndarray 형식의 이미지 픽셀 정보를 이미지 파일로 저장할 수 있습니다.
"""
import numpy as np
from PIL import Image

from ch03.ex06 import img_show


def image_to_pixel(image_file):
    """이미지 파일 이름(경로)를 파라미터로 전달받아서,
    numpy.ndarray에 픽셀 정보를 저장해서 리턴."""
    img = Image.open(image_file)
    # img.show()
    array = np.array(img)
    return array


def pixel_to_image(pixel, image_file):
    """numpy.ndarray 형식의 이미지 픽셀 정보와, 저장할 파일 이름을 파라미터로
    전달받아서, 이미지 파일을 저장"""
    img = Image.fromarray(pixel)
    img.save(image_file)


if __name__ == '__main__':
    # image_to_pixel(), pixel_to_image() 함수 테스트
    img = image_to_pixel('C:/Users/user/Desktop/patrick.jpg')
    img_show(img)
    pixel_to_image(img, 'patrick.jpg')
