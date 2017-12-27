import scipy.misc
import libs.stereo_pair as STEREO_PAIR
import libs.frame as FRAME
import libs.camera_model as CAMERA

"""Given two images (key and reference frame) loads them into memory from their directory.
"""


class StereoPairBuilder:
    def __init__(self, camera_model, image_loader, odometry, resize, runtime_settings):
        self.camera_model = camera_model
        self.image_loader = image_loader
        self.odometry = odometry
        self.key_frame = None
        self.runtime_settings = None
        self.resize = resize
        self.runtime_settings = runtime_settings

    def make_key_frame(self, key):
        if (self.key_frame is None):
            pic_rect_key = self.image_loader.load(key, self.camera_model)
            pic_key_resize = pic_rect_key
            if(self.resize > 0):
                pic_key_resize = scipy.misc.imresize(pic_rect_key, self.resize)

            key_frame = FRAME.Frame(pic_key_resize, key)
            self.key_frame = key_frame

        return self.key_frame

    def generate_stereo_pairs(self, key, reference_frame_list,invert=False):

        stereo_pairs = []

        key_frame = self.make_key_frame(key)
        K_key = self.camera_model.build_K_matrix()
        if len(self.camera_model.intrinsics) > 0:
            K_key = self.camera_model.build_K_matrix_from_id(key)

        for reference_frame in reference_frame_list:
            pic_rect_ref = self.image_loader.load(reference_frame, self.camera_model)

            pic_ref_resize = pic_rect_ref
            if (self.resize > 0):
                pic_ref_resize = scipy.misc.imresize(pic_rect_ref, self.resize)
            (height, width) = (pic_rect_ref.height, pic_rect_ref.width)

            ref_frame = FRAME.Frame(pic_ref_resize, reference_frame)

            se3 = None

            if invert:
                se3 = self.odometry.buildSE3(reference_frame, key)
            else:
                se3 = self.odometry.buildSE3(key, reference_frame)

            K_ref = K_key
            if len(self.camera_model.intrinsics) > 0:
                K_ref = self.camera_model.build_K_matrix_from_id(reference_frame)

            stereo_pair = STEREO_PAIR.StereoPair(key_frame, ref_frame, se3, K_key,K_ref, invert_se3_matricies=invert)

            stereo_pairs.append(stereo_pair)

        return stereo_pairs
