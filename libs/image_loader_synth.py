from PIL import Image

"""Loads an Image from the Scaramuzza Dataset
"""

class ImageLoader:

    # TODO refector to work with different camera loading models
    def __init__(self,images_dir):
        self.images_dir = images_dir


    def load(self,image_id,camera_model=''):
        # img_id_str = str(image_id)
        # amount_of_zeros = 4-len(img_id_str)
        #
        # img_prefix = 'img'
        #
        # for i in range(0,amount_of_zeros):
        #     img_prefix = img_prefix + '0'

        img_path = self.images_dir + '/' + ImageLoader.parse_id(image_id) + '.png'
        return Image.open(img_path)

    @staticmethod
    def parse_id(image_id):
        img_id_str = str(image_id)
        amount_of_zeros = 4 - len(img_id_str)

        img_prefix = 'img'

        for i in range(0, amount_of_zeros):
            img_prefix = img_prefix + '0'

        return img_prefix + img_id_str + '_0'



