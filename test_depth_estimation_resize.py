import libs.sdk.python.image
import libs.sdk.python.camera_model
from PIL import Image
import scipy.misc

import libs.vo_parser as vo_parser
import libs.stereo_pair
import libs.frame
import libs.settings
import depth_estimation as depth_est
import libs.visualize

models_dir = 'datasets/small_1/camera-models/'
images_dir = 'datasets/small_1/frames/stereo/centre'

# Pair 1


#source = 1400075966431698
#dest = 1400075966369207

# Pair 2
#source = 1400076022361589
#dest = 1400076022299089

# Pair 3
#source = 1400076019861975
#dest = 1400076019799486

# Pair 4 - best

#source = 1400076019049605
#dest = 1400076018987109

# Pair 5
#source = 1400076018862131
#dest = 1400076018799634

#Pair 6

#dest = 1400076116598857
#source = 1400076116661344

# Pair 7

dest = 1400076139345758
source = 1400076139408250


img_path_dest = images_dir + '/' + str(dest) + '.png'
img_path_source = images_dir + '/' + str(source) + '.png'

cm = libs.sdk.python.camera_model.CameraModel(models_dir, images_dir)

data_rect_dest = libs.sdk.python.image.load_image(img_path_dest, model=cm)
data_dest = libs.sdk.python.image.load_image(img_path_dest)

pic_rect_dest = Image.fromarray(data_rect_dest, 'RGB')
pic_dest = Image.fromarray(data_dest, 'RGB')

data_rect_source = libs.sdk.python.image.load_image(img_path_source, model=cm)
data_source = libs.sdk.python.image.load_image(img_path_source)
pic_rect_source = Image.fromarray(data_rect_source, 'RGB')

vo = vo_parser.VoParser('datasets/small_1/vo/vo.csv')
vo_all = vo.load_csv()

## Resize ##

pic_dest_resize = scipy.misc.imresize(pic_rect_dest, 25)
pic_source_resize = scipy.misc.imresize(pic_rect_source, 25)

##

(height_re, width_re, channels) = pic_source_resize.shape
(height, width) = (pic_dest.height, pic_dest.width)
# runtime_settings = libs.settings.Settings(cm.focal_length[0],cm.focal_length[1],cm.principal_point[0],cm.principal_point[1],height,width)

runtime_settings_re = libs.settings.Settings(cm.focal_length[0], cm.focal_length[1], cm.principal_point[0],
                                             cm.principal_point[1], height_re, width_re)

runtime_settings = libs.settings.Settings(cm.focal_length[0], cm.focal_length[1], cm.principal_point[0],
                                          cm.principal_point[1], height, width)

# source_frame = libs.frame.Frame(pic_rect_source,source)
# dest_frame = libs.frame.Frame(pic_rect_dest,dest)

dest_frame_re = libs.frame.Frame(pic_source_resize, source)
source_frame_re = libs.frame.Frame(pic_dest_resize, dest)

se3 = vo.buildSE3(source, dest)
K = runtime_settings.build_K_matrix()

stereo_pair_re = libs.stereo_pair.StereoPair(source_frame_re, dest_frame_re, se3, K, invert_se3_matricies=True)
depth_estimation_re = depth_est.DepthEstimation(stereo_pair_re, runtime_settings)

## Plot Initial Values ##

# libs.visualize.show(stereo_pair.key_frame.get_depth_map(),normalize=True)
# libs.visualize.show(stereo_pair.key_frame.get_image())
# libs.visualize.show_prob_custom(stereo_pair.key_frame.get_depth_map(),0.4999,0.50001)



## Visualize Good Pixels For Resized ##

# (successes, good_pixels) = depth_estimation_re.search_for_good_pixels()
# libs.visualize.visualize_keypoints(stereo_pair_re.key_frame.get_image(),good_pixels)

## Visualize All Pixels For Resized ##

(successes, matches,matches_for_large_disparities_in_x,marked_keypoints) = depth_estimation_re.depth_estimation()
libs.visualize.visualize_successes_and_discarded_pixels(stereo_pair_re.key_frame.get_image(),marked_keypoints)


## 1 iteration of depth estimation for Resized ##

libs.visualize.show(stereo_pair_re.key_frame.get_depth_map(),normalize=True)
# libs.visualize.show_prob_custom(stereo_pair_re.key_frame.get_depth_map(),0.4999,0.50001)

## Visualize Matching for Resized

# match_amount = len(matches)
# perc = 0.05
# idx = int(perc*match_amount)
# (x,y, x_match,y_match) = matches[idx]
# ps = [(x,y)]
# match_ps = [(x_match,y_match)]

# libs.visualize.visualize_matches(stereo_pair_re.key_frame.get_image(),stereo_pair_re.reference_frame.get_image(),(x,y),(x_match,y_match))

# libs.visualize.visualize_matches(stereo_pair_re.key_frame.get_image(),stereo_pair_re.reference_frame.get_image(),(100,100),(100,100))

# Visualize Epipolar Lines
# match_amount = len(epipolar_matches)
# perc = 0.05
# idx = int(perc*match_amount)
# (x,y,epx,epy,p_close_ref,p_far_ref) = epipolar_matches[idx]

# p_close_ref_x = p_close_ref[0,0]
# p_close_ref_y = p_close_ref[1,0]
# p_far_ref_x = p_far_ref[0,0]
# p_far_ref_y = p_far_ref[1,0]

# p_key = (x,y,epx,epy)
# p_ref = (p_close_ref_x,p_close_ref_y,p_far_ref_x,p_far_ref_y)

# libs.visualize.visualize_epipolar_lines(stereo_pair_re.key_frame.get_image(),stereo_pair_re.reference_frame.get_image(),p_key,p_ref)



# Visualize Matches and Epipolar Line
# match_amount = len(matches)
# perc = 0.90
#
# idx = int(perc * match_amount)
# (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref) = matches[idx]
#
# p_close_ref_x = p_close_ref[0, 0]
# p_close_ref_y = p_close_ref[1, 0]
# p_far_ref_x = p_far_ref[0, 0]
# p_far_ref_y = p_far_ref[1, 0]
#
# ep_ref = (p_close_ref_x, p_close_ref_y, p_far_ref_x, p_far_ref_y)
#
# libs.visualize.visualize_match_and_epipolar(stereo_pair_re.key_frame.get_image(),
#                                             stereo_pair_re.reference_frame.get_image(), (x, y), (match_x, match_y),
#                                             ep_ref)
