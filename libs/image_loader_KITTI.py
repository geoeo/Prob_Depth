from PIL import Image

"""Loads an Image from the KITTI Dataset
"""

class ImageLoader:

    def __init__(self,images_dir):
        self.images_dir = images_dir

    def load(self, image_id,camera_model=''):
        img_id_str = str(image_id)
        amount_of_zeros = 6 - len(img_id_str)

        img_prefix= ''

        for i in range(0,amount_of_zeros):
            img_prefix = img_prefix + '0'

        img_str = img_prefix + img_id_str

        img_path = self.images_dir + img_str + '.png'
        return Image.open(img_path)







