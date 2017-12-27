import libs.sdk.python.image
from PIL import Image

"""Loads an Image; Given a Camera Model, undistorts it as well
"""

class ImageLoader:

    # TODO refector to work with different camera loading models
    def __init__(self,images_dir):
        self.images_dir = images_dir

    def load(self,image_id,camera_model):
        img_path = self.images_dir + '/' + str(image_id) + '.png'
        data_rect = libs.sdk.python.image.load_image(img_path, model=camera_model)
        return Image.fromarray(data_rect, 'RGB')

    def load_rectified(self,image_id,camera_model=''):
        img_path = self.images_dir + '/' + str(image_id) + '.png'
        return Image.open(img_path, 'RGB')

