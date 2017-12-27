import numpy as np

"""A Data Object which encodes the extrinsic state of the camera.

Attributes:
  se3: the matrix which encodes rotation and translation
  se3_inv: the inverse matrix
"""


class ExtrinsicCamera:
    def __init__(self, se3, invert_matricies=False):
        self.se3 = se3
        self.se3_inv = ExtrinsicCamera.inverse(se3)

        if(invert_matricies):
            self.se3 = self.se3_inv
            self.se3_inv = se3


    @staticmethod
    def extract_rotation(se3):
        return se3[0:3,0:3]

    @staticmethod
    def extract_translation(se3):
        return se3[0:3,3:4]

    @staticmethod
    def extract_r_t(se3):
        # returns 3x4 sub matrix
        return se3[0:3,0:4]

    @staticmethod
    def inverse(se3):
        rotation = ExtrinsicCamera.extract_rotation(se3)
        rotation_transpose = np.transpose(rotation)

        translation = ExtrinsicCamera.extract_translation(se3)
        translation_inverse = -1 * translation

        m = np.concatenate((rotation_transpose, translation_inverse), axis=1)
        se3_inv = ExtrinsicCamera.append_homogeneous_along_y(m)

        return se3_inv

    @staticmethod
    def append_homogeneous_along_y(m):
        return np.concatenate((m, np.array([[0, 0, 0, 1]])), axis=0)
