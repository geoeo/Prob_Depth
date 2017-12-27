import itertools
import numpy as np
import pykitti

class VoParserSynth:
    def __init__(self, path,sequence):
        self.vo_path = path # base_dir
        self.sequence = sequence

    def buildSE3(self, sourceImageId, destImageId):
        image_range = range(sourceImageId, destImageId, 1)
        dataset = pykitti.odometry(self.vo_path, self.sequence, frames=image_range)

        SE3_i = next(itertools.islice(dataset.poses, sourceImageId, None))
        SE3_j = next(itertools.islice(dataset.poses, destImageId, None))

        R_i_trans = np.transpose(SE3_i[0:3,0:3])
        R_ij = np.matmul(SE3_j[0:3,0:3],R_i_trans)

        t_i_c = np.matmul(R_i_trans,SE3_i[0:3,3])
        t_j_c = np.matmul(R_i_trans,SE3_j[0:3,3])
        t_ij = t_j_c - t_i_c

        # t_ij = np.matmul(SE3_i[0:3,0:3],t_ij) # Account for rotation of the camera
        # t_ij[2] *= -1
        se3 = np.identity(4)
        se3[0:3,0:3] = R_ij
        se3[0:3,3] = t_ij

        return se3




