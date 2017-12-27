"""Given a Keyframe, a list of Reference Frames and Camera Data builds Stereo Pairs.
"""
import libs.sdk.python.camera_model

class CameraModelLoader:

    #TODO refector to work with different camera loading models
    def __init__(self,models_dir,cam_type):
        self.models_dir = models_dir
        self.cam_type = cam_type

    def load(self):
        return libs.sdk.python.camera_model.CameraModel(self.models_dir, self.cam_type)