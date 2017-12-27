import libs.sdk.python.image
import libs.sdk.python.camera_model
import re
import os
from PIL import Image
import numpy as np

import libs.vo_parser
import libs.stereo_pair
import libs.frame
import libs.sdk.python.interpolate_poses
import depth_estimation
import libs.settings

models_dir = 'datasets/small_1/camera-models/'
#images_dir = 'datasets/ox/sample_small/mono_left'
#img_path = 'datasets/ox/sample_small/mono_left/1418381798213163.png'

images_dir = 'datasets/small_1/frames/stereo/centre'
vo_file = 'datasets/small_1/vo/vo.csv'

source = 1400075966431698
dest = 1400075966369207

img_path_dest = images_dir+ '/' + str(dest) + '.png'
img_path_source = images_dir + '/' + str(source) +'.png'

cm = libs.sdk.python.camera_model.CameraModel(models_dir,images_dir)

data_rect_dest = libs.sdk.python.image.load_image(img_path_dest,model=cm)
data_dest = libs.sdk.python.image.load_image(img_path_dest)

pic_rect_dest = Image.fromarray(data_rect_dest, 'RGB')
pic_dest = Image.fromarray(data_dest, 'RGB')

(height,width) = (pic_dest.height,pic_dest.width)

runtime_settings = libs.settings.Settings(cm.focal_length[0],cm.focal_length[1],cm.principal_point[0],cm.principal_point[1],height,width)


data_rect_source = libs.sdk.python.image.load_image(img_path_source,model=cm)
data_source = libs.sdk.python.image.load_image(img_path_source)
pic_rect_source = Image.fromarray(data_rect_source, 'RGB')

vo = libs.vo_parser.VoParser(vo_file)
vo_all = vo.load_csv()

# -------------------------- #

#pic_rect.show()
#pic.show()
#pic_rect_2.show()

 # -------------------------- #

# print(vo_all[1][0])

# ------------------------- #

#stereo_pair = libs.stereo_pair.StereoPair(source,pic_rect_2,dest,pic_rect,[])
#stereo_pair.currentFrame.show()
#stereo_pair.refFrame.show()

#frame = libs.frame.Frame(pic_rect_source,source)

#stereo_pair.calcImageGradientInCurrentLap()
#stereo_pair.calcImageGradientInCurrentSobelX()
#stereo_pair.calcImageGradientInCurrentSobelY()

#print(frame.width,frame.height)
#print(frame.image[0][0])
#frame.calcImageGradientInCurrentLap()
#frame.calcImageGradientInCurrentLap()
#grad_x = frame.calcImageGradientSimpleX(0,1)
#grad_y = frame.calcImageGradientSimpleY(0,1)
#print((grad_x,grad_y))

# ------------------------ #

#dest = 1400075966306717

#print(vo.vo_data[66])

se3 = vo.buildSE3(source,dest)
vo_path =  'datasets/ox/2014-05-14-13-59-05/vo/vo.csv'
se3_list = libs.sdk.python.interpolate_poses.interpolate_vo_poses(vo_path,[dest],source)
print(se3)
print(se3_list[0])

# ------------------------ #

#source_frame = libs.frame.Frame(pic_rect_source,source)
#dest_frame = libs.frame.Frame(pic_rect_dest,dest)
#se3 = vo.buildSE3(source,dest)
#K = runtime_settings.build_K_matrix()

#stereo_pair = libs.stereo_pair.StereoPair(dest_frame,source_frame,se3,K)
#depth_estimation = depth_estimation.DepthEstimation(stereo_pair,runtime_settings)

#libs.frame.Frame.show(depth_estimation.reference_frame.get_absolute_gradient())
#libs.frame.Frame.show(depth_estimation.key_frame.get_absolute_gradient())

#depth_estimation.search_for_good_pixels()







