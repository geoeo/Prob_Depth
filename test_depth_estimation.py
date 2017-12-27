import libs.sdk.python.image
import libs.sdk.python.camera_model
import re
import os
from PIL import Image
import numpy as np
import scipy.misc

import libs.vo_parser
import libs.stereo_pair
import libs.frame
import depth_estimation as depth_est
import libs.settings
import libs.visualize

models_dir = 'datasets/small_1/camera-models/'
images_dir = 'datasets/small_1/frames/stereo/centre'

#Pair 1
#source = 1400075966431698
#dest = 1400075966369207

# Pair 2
source = 1400076022361589
dest = 1400076022299089

img_path_dest = images_dir+ '/' + str(dest) + '.png'
img_path_source = images_dir + '/' + str(source) +'.png'

cm = libs.sdk.python.camera_model.CameraModel(models_dir,images_dir)

data_rect_dest = libs.sdk.python.image.load_image(img_path_dest,model=cm)
data_dest = libs.sdk.python.image.load_image(img_path_dest)

pic_rect_dest = Image.fromarray(data_rect_dest, 'RGB')
pic_dest = Image.fromarray(data_dest, 'RGB')

data_rect_source = libs.sdk.python.image.load_image(img_path_source,model=cm)
data_source = libs.sdk.python.image.load_image(img_path_source)
pic_rect_source = Image.fromarray(data_rect_source, 'RGB')

vo = libs.vo_parser.VoParser('datasets/small_1/vo/vo.csv')
vo_all = vo.load_csv()

## Resize ##

# pic_dest_resize = scipy.misc.imresize(pic_rect_dest,25)
# pic_source_resize = scipy.misc.imresize(pic_rect_source,25)


(height,width) = (pic_dest.height,pic_dest.width)

runtime_settings = libs.settings.Settings(cm.focal_length[0],cm.focal_length[1],cm.principal_point[0],cm.principal_point[1],height,width)

source_frame = libs.frame.Frame(pic_rect_source,source)
dest_frame = libs.frame.Frame(pic_rect_dest,dest)

# source_frame_re = libs.frame.Frame(pic_source_resize,source)
# dest_frame_re = libs.frame.Frame(pic_dest_resize,dest)

se3 = vo.buildSE3(source,dest)
K = runtime_settings.build_K_matrix()

stereo_pair = libs.stereo_pair.StereoPair(dest_frame,source_frame,se3,K)
depth_estimation = depth_est.DepthEstimation(stereo_pair,runtime_settings)


## Plot Initial Values ##

#libs.visualize.show(stereo_pair.key_frame.get_depth_map(),normalize=True)
#libs.visualize.show(stereo_pair.key_frame.get_image())
#libs.visualize.show_prob_custom(stereo_pair.key_frame.get_depth_map(),0.4999,0.50001)

## Visualize Good Pixels ##

#(successes, good_pixels) = depth_estimation.search_for_good_pixels()
#libs.visualize.visualize_keypoints(stereo_pair.key_frame.get_image(),good_pixels)

## 1 iteration of depth estimation ##

#(successes, matches,epipolar_matches) = depth_estimation.depth_estimation()
#libs.visualize.show(stereo_pair.key_frame.get_depth_map(),normalize=True)

## Visualize Matching for Resized

#(successes, matches,epipolar_matches) = depth_estimation.depth_estimation()

#(x,y, x_match,y_match) = matches[3]

#ps = [(x,y)]
#match_ps = [(x_match,y_match)]

#libs.visualize.visualize_matches(stereo_pair.key_frame.get_image(),stereo_pair.reference_frame.get_image(),(x,y),(x_match,y_match))

#libs.visualize.visualize_matches(stereo_pair.key_frame.get_image(),stereo_pair.reference_frame.get_image(),(325,334),(370,380))









