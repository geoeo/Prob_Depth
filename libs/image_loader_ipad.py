from PIL import Image

"""Loads an Image from the ipad Dataset
"""

class ImageLoader:

    def __init__(self,images_dir):
        self.images_dir = images_dir

    def load(self, image_id,camera_model=''):
        img_id_str = 'frame_' + str(image_id)

        img_path = self.images_dir + img_id_str + '.png'
        return Image.open(img_path)







