"""Settings for the stereo algorithm.
    Values taken from lsd_slam_core/src/util/settings.h
"""

import numpy as np


"""Runtime setting which depend on other factors e.g. camera used
  """


class Settings:
    def __init__(self,min_epl_grad,min_epl_length_sq,min_epl_angle,epl_scale,min_depth, invert_depth):
        self.min_epl_grad = min_epl_grad
        self.min_epl_length_sq = min_epl_length_sq
        self.min_epl_angle = min_epl_angle
        self.epl_scale = epl_scale
        self.min_depth = min_depth
        self.max_depth = 1/min_depth
        self.invert_depth = invert_depth



    @staticmethod
    def max_epl_length_crop():
        return 30

    @staticmethod
    def min_epl_length_crop():
        return 3

    @staticmethod
    def min_abs_grad_decrease():
        return 1  # original value is 5

    @staticmethod
    def min_abs_grad_create():
        return 1  # original value is 5

    @staticmethod
    def sample_point_to_border():
        return 7

    def min_epl_grad_squared(self):
        return self.min_epl_grad * self.min_epl_grad  # 2 - original

    def min_epl_length_squared(self):
        # return 1.0*1.0
        return self.min_epl_length_sq  # 1 - original

    def min_epl_angle_squared(self):
        return self.min_epl_angle * self.min_epl_angle  # 0.3 orginal

    def get_epl_scale(self):
        return self.epl_scale

    def get_min_depth(self):
        return self.min_depth # 0.05

    @staticmethod
    def gradient_sample_dist():
        return 1.0  # 1.0 original

    @staticmethod
    def succ_var_inc_fac():
        return 1.01

    # maximal photometric error for stereo to be successful (sum over 5 squared intensity differences)
    @staticmethod
    def max_error_stereo():
        return 1300.0 # 1300.0

    # defines how large the stereo-search region is. it is [mean] +/- [std.dev]*STEREO_EPL_VAR_FAC
    @staticmethod
    def stereo_epl_var_fac():
        return 2.0

    # minimal multiplicative difference to second-best match to not be considered ambiguous.
    @staticmethod
    def min_distance_error_stereo():
        return 1.5

    @staticmethod
    def camera_pixel_noise():
        return 4.0 * 4.0  # orig 4*4

    @staticmethod
    def max_var():
        return 0.5 * 0.5

    @staticmethod
    def var_random_init():
        return 0.5 * Settings.max_var()

    @staticmethod
    def division_eps():
        return 1e-10

    # minimal summed validity over 5x5 region to keep hypothesis (regularization)
    @staticmethod
    def val_sum_min_for_keep():
        return  5 # orig 24 (with a 5 increase for a valid pixel)

    # minimal summed validity over 5x5 region to create a new hypothesis for non-blacklisted pixel (hole-filling)
    @staticmethod
    def min_val_for_fill():
        return 10

    @staticmethod
    def reg_dist_var():
        return 0.075*0.075*1.0*1.0 # last term is depth smoothing factor

    @staticmethod
    def UNZERO(val):

        clamped = val
        min = -1e-10
        max = 1e-10

        if val < 0:
            if val > min:
                clamped = min
        elif val < max:
            clamped = max

        return clamped
