import numpy as np

"""A Data Object which encodes the intrinsic state of the camera.

Attributes:
  K: the matrix which encodes the intrinsic camera parameters: focal length and principal offset 3x4 Matrix
  K_inv: the inverse matrix 3x3 Matrix
"""


class IntrinsicCamera:
    def __init__(self, K):
        self.K = K
        self.K_inv = IntrinsicCamera.inverse(K)

    @staticmethod
    def extract_fx(K):
        return K[0, 0]

    @staticmethod
    def extract_fy(K):
        return K[1, 1]

    @staticmethod
    def extract_cx(K):
        return K[0, 2]

    @staticmethod
    def extract_cy(K):
        return K[1, 2]

    # Returns 3x3 Intrinsic Camera Matrix
    @staticmethod
    def inverse(K):
        fx = IntrinsicCamera.extract_fx(K)
        fy = IntrinsicCamera.extract_fy(K)
        fx_inv = 1 / fx
        fy_inv = 1 / fy
        cx_inv = -1 * IntrinsicCamera.extract_cx(K) / fx
        cy_inv = -1 * IntrinsicCamera.extract_cy(K) / fy

        K_inv = np.array([[fx_inv, 0, cx_inv],
                          [0, fy_inv, cy_inv],
                          [0, 0, 1]])
        # K_inv = np.linalg.inv(K[0:3,0:3])

        return K_inv
