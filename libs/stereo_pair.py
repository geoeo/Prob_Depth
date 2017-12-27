import numpy as np
import libs.extrinsic_camera as extrinsic_cam
import libs.intrinsic_camera as intrinsic_cam

"""A Data Object with two Images and a corresponding SE3 matrix.
Images are already undistorted and converted to grey values.

Attributes:
  reference_frame: The image we transform to, with the SE3 matrix
  key_frame: The image we consider our origin -  - frame we compute the depth estimate for (current)
  extrinsic: DTO - The transformation matrix between reference and key_frame
  intrinsic: Matrix describing the intrinsic camera parameters
  k_ref_to_key: Matrix pipeline - intrinsic * extrinsic.se3_inv
  k_key_to_ref: Matrix pipeline - intrinsic * extrinsic.se3
"""


class StereoPair:
    def __init__(self, key_frame, ref_frame, se3,k_key,k_ref, invert_se3_matricies=False):
        self.key_frame = key_frame
        self.reference_frame = ref_frame
        self.intrinsic = intrinsic_cam.IntrinsicCamera(k_key)
        self.intrinsic_ref = intrinsic_cam.IntrinsicCamera(k_ref)

        self.extrinsic = extrinsic_cam.ExtrinsicCamera(se3, invert_se3_matricies)

        self.ref_to_key = self.extrinsic.se3_inv
        self.key_to_ref = self.extrinsic.se3

        self.k_ref_to_key_rotation = np.matmul(self.intrinsic.K,StereoPair.extract_rotation(self.extrinsic.se3_inv)) # 3x3 matrix
        self.k_key_to_ref_rotation = np.matmul(self.intrinsic_ref.K,StereoPair.extract_rotation(self.extrinsic.se3)) # 3x3 matrix

        self.k_ref_to_key_translation = np.matmul(self.intrinsic.K,StereoPair.extract_translation(self.extrinsic.se3_inv)) # 3x3 matrix
        self.k_key_to_ref_translation = np.matmul(self.intrinsic_ref.K,StereoPair.extract_translation(self.extrinsic.se3)) # 3x3 matrix

    @staticmethod
    def extract_rotation(se3):
        return extrinsic_cam.ExtrinsicCamera.extract_rotation(se3)

    @staticmethod
    def extract_translation(se3):
        return extrinsic_cam.ExtrinsicCamera.extract_translation(se3)

    @staticmethod
    def extract_r_t(se3):
        # returns 3x4 sub matrix
        return extrinsic_cam.ExtrinsicCamera.extract_r_t(se3)

    def return_k_ref_to_key_rotation(self):
        return self.k_ref_to_key_rotation

    def return_k_ref_to_key_translation(self):
        return self.k_ref_to_key_translation

    def return_k_key_to_ref_rotation(self):
        return self.k_key_to_ref_rotation

    def return_k_key_to_ref_translation(self):
        return self.k_key_to_ref_translation

    def return_ref_to_key_rotation(self):
        return StereoPair.extract_rotation(self.ref_to_key)

    def return_ref_to_key_translation(self):
        return StereoPair.extract_translation(self.ref_to_key)

    def return_key_to_ref_rotation(self):
        return StereoPair.extract_rotation(self.key_to_ref)

    def return_key_to_ref_translation(self):
        return StereoPair.extract_translation(self.key_to_ref)

    def return_width(self):
        return self.reference_frame.width

    def return_height(self):
        return self.reference_frame.height

    def return_fx(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_fx(intrinsic.K)

    def return_fy(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_fy(intrinsic.K)

    def return_cx(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_cx(intrinsic.K)

    def return_cy(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_cy(intrinsic.K)

    def return_fxi(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_fx(intrinsic.K_inv)

    def return_fyi(self,intrinsic):
        return intrinsic_cam.IntrinsicCamera.extract_fy(intrinsic.K_inv)
