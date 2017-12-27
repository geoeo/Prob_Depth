import numpy as np
import math
import csv
import os


class Evaluation:

    @staticmethod
    def calc_avg_mse(root_dir):
        min_list = []

        for subdir, dirs, files in os.walk(root_dir):
            try:
                file_path = os.path.join(subdir,'mse_list.txt')
                data = Evaluation.load_data_list(file_path)
                min_list.append(np.amin(data))
            except Exception:
                pass #ignore root dir

        return np.average(min_list)


    @staticmethod
    def load_ground_truth(file,rows,cols,flip_across_y = False):
        A = np.resize(np.loadtxt(file),(rows,cols))
        if flip_across_y:
            A = np.flip(A,1)
        return A

    @staticmethod
    def load_data_list(file_path):
        file = open(file_path, 'r')
        reader = csv.reader(file, delimiter=' ')
        data_list = []

        for row in reader:
            for data_raw in row:
                if not data_raw.isspace() and data_raw:
                    data_list.append(float(data_raw))

        return data_list


    @staticmethod
    def calc_inverse_ground_truth(ground_truth, runtime_settings, focal_length = -1, max_thresh = 200,isIpad = False):
        post_processed_gt = ground_truth
        if focal_length is not -1:
            post_processed_gt = Evaluation.convert_matrix_to_depth_along_z(post_processed_gt,focal_length)
        # ground_truth = ground_truth_euclid
        # post_processed_gt = Evaluation.threshold(post_processed_gt, 1e-2, max_thresh)
        post_processed_gt = Evaluation.invert(post_processed_gt, 1e-2,isIpad)
        post_processed_gt = Evaluation.scale_matrix(post_processed_gt, runtime_settings.min_depth,
                                                         runtime_settings.max_depth)
        return post_processed_gt

    @staticmethod
    def convert_matrix_to_depth_along_z(input_matrix,focal_length):
        (height, width) = input_matrix.shape
        output = np.empty((height,width))
        for r in range(0,height):
            for c in range(0,width):
                depth_euclid = input_matrix[r,c]
                # x = c
                # y = r
                # # center bottom middle
                # x = c - 320
                # y = 480  - r
                # center middle
                x = c - width/2 # principal point x
                y = r - height/2 # principal point y
                #https: // www.doc.ic.ac.uk / ~ahanda / VaFRIC / codes.html
                f_z = focal_length * math.sqrt((depth_euclid*depth_euclid)/(x*x+y*y+focal_length*focal_length))
                output[r,c] = f_z

        return output

    @staticmethod
    def threshold(input_matrix,thresh_min,thresh_max):
        out = input_matrix.copy()
        # input = input_matrix
        # mask = input_matrix == 0
        # out = np.ma.array(input, mask=mask)
        (height, width) = input_matrix.shape
        for y in range(0, height, 1):
            for x in range(0, width, 1):
                if input_matrix[y,x] == 0: continue
                val = input_matrix[y, x]
                if (val > thresh_max):
                    val = thresh_max  # clamp value to max_depth
                elif val < thresh_min:
                    val = thresh_min
                out[y, x] = val

        return out

    @staticmethod
    def invert(input_matrix,thresh_min,isIpad):
        out = input_matrix.copy()
        # mask = input_matrix == 0
        # out = np.ma.array(input_matrix, mask=mask)
        (height, width) = input_matrix.shape
        # out = 1 / out
        for y in range(0, height, 1):
            for x in range(0, width, 1):
                if input_matrix[y,x] == 0: continue
                val = input_matrix[y, x]
                out[y, x] = 1/val
        # rescale
        max = np.amax(out)
        min = np.amin(out)
        diff = 0
        if not isIpad:
            diff = min - thresh_min
        # scale
        out = out - diff
        return out

    @staticmethod
    def scale_matrix(input_matrix,scale_min,scale_max,hypothesis_map=None):
        (height, width) = input_matrix.shape
        if hypothesis_map is not None:
            out = np.ma.array(input_matrix, mask=~hypothesis_map)
        else:
            out = input_matrix.copy()

        max = np.amax(out)
        min = np.amin(out)
        scale = scale_max - scale_min

        # for y in range(0, height, 1):
        #     for x in range(0, width, 1):
        #         if input_matrix[y,x] == 0: continue
        #         if hypothesis_map is not None:
        #             if not hypothesis_map[y,x]:
        #                 continue
        #         val = scale*out[y,x]/max
        #         out[y, x] = val+scale_min


        # out = out - min
        out = scale*out/max
        out += scale_min
        return out

    @staticmethod
    def set_zero_to_min(input_matrix,min):
        out = input_matrix.copy()
        (height, width) = input_matrix.shape
        for r in range(0, height):
            for c in range(0, width):
                if input_matrix[r,c] == 0:
                    out[r,c] = min

        return out

    @staticmethod
    def compute_rms(ground_truth,depth_map,hypothesis_map):

        #  if no depth values exist
        if not hypothesis_map.max():
            return -1.0

        (height, width) = ground_truth.shape
        mse = 0
        count = 0
        for r in range(0, height, 1):
            for c in range(0, width, 1):
                if hypothesis_map[r,c] and ground_truth[r,c] != 0:
                    count += 1
                    error = math.fabs(ground_truth[r,c] - depth_map[r,c])
                    mse += error*error
        mse /= count
        mse = math.sqrt(mse)
        return mse





