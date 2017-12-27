import libs.vo_parser as VO_PARSER
import libs.camera_model_loader as CAMERA_MODEL_LOADER
import libs.image_loader as IMAGE_LOADER
import libs.stereo_pair_builder as STEREO_PAIR_BUILDER
import depth_estimation as DEPTH_ESTIMATION
import libs.settings as SETTINGS
import libs.visualize


root_dir = 'datasets/iterative/'
models_dir = root_dir+'camera-models/'
cam_type = 'stereo/centre'
images_dir = root_dir+'frames/stereo/centre'
odom_path = 'datasets/small_1/vo/vo.csv'

# Pair 4 - best

# ref_5 = 1400076019299568
# ref_4 = 1400076019237077
ref_3 = 1400076019174586
ref_2 = 1400076019112097
ref_1 = 1400076019049605
key = 1400076018987109 # keyframe

# ref_7 = 1400076117223752
# ref_6 = 1400076117161262
# ref_5 = 1400076117098774
# ref_4 = 1400076117036283
# ref_3 = 1400076116973793
# ref_2 = 1400076116911304
# ref_1 = 1400076116848813
# key = 1400076116786324 # keyframe

# ref_6 = 1400076139533227
# ref_5 = 1400076139470738
# ref_4 = 1400076139408250
# ref_3 = 1400076139345758
# ref_2 = 1400076139283268
# ref_1 = 1400076139220778
# key = 1400076139158288


camera_loader = CAMERA_MODEL_LOADER.CameraModelLoader(models_dir,cam_type)
cm = camera_loader.load()
image_loader = IMAGE_LOADER.ImageLoader(images_dir)
vo = VO_PARSER.VoParser(odom_path)
runtime_settings = SETTINGS.Settings(cm.focal_length[0],
                                     cm.focal_length[1],
                                     cm.principal_point[0],
                                     cm.principal_point[1],
                                     12, 0.01, 0.3, 1.0)

ref_list = [ref_1,ref_2,ref_3]

stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 25, runtime_settings)
stereo_pairs = stereo_pair_builder.generate_stereo_pairs(key, ref_list, invert=True)

iteration = 1
for stereo_pair in stereo_pairs:
    #libs.visualize.show_frame(stereo_pair.key_frame.get_image(),root_dir,'keyframe')
    it_string = str(iteration)
    depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, stereo_pair_builder.runtime_settings)
    (successes, matches,matches_for_large_disparities_in_x,marked_keypoints) = depth_estimation_re.depth_estimation(True)
    libs.visualize.show(stereo_pair_builder.key_frame.get_depth_map(),
                        stereo_pair_builder.key_frame.get_hypothesis_map(), normalize=True, iteration=it_string,
                        show=False, path=root_dir, color_map='nipy_spectral')
    #libs.visualize.show(stereo_pair_builder.key_frame.get_depth_map(),normalize=True,iteration=it_string,show=False,path=root_dir,color_map='gray')
    # libs.visualize.show_color(stereo_pair_builder.key_frame.get_depth_map(),stereo_pair_builder.key_frame.get_hypothesis_map(), iteration=it_string, path=root_dir)
    # libs.visualize.show_color_with_keyframe(stereo_pair_builder.key_frame.get_image(),stereo_pair_builder.key_frame.get_depth_map(),stereo_pair_builder.key_frame.get_hypothesis_map(), iteration=it_string, path=root_dir)
    iteration = iteration + 1

# for stereo_pair in stereo_pairs:
    # Visualize Matches and Epipolar Line
# [stereo_1] = stereo_pairs
# depth_estimation_1 = DEPTH_ESTIMATION.DepthEstimation(stereo_1, stereo_pair_builder.runtime_settings)
# (successes, matches, matches_for_large_disparities_in_x, marked_keypoints) = depth_estimation_1.depth_estimation()
# # depth_estimation_2 = DEPTH_ESTIMATION.DepthEstimation(stereo_2, stereo_pair_builder.runtime_settings)
# # (successes, matches, matches_for_large_disparities_in_x, marked_keypoints) = depth_estimation_2.depth_estimation()
# match_amount = len(matches)
#
# for idx in range(0,match_amount,200):
#     (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref,rescale_factor,K_t_vector) = matches[idx]
#
#     p_close_ref_x = p_close_ref[0, 0]
#     p_close_ref_y = p_close_ref[1, 0]
#     p_far_ref_x = p_far_ref[0, 0]
#     p_far_ref_y = p_far_ref[1, 0]
#
#     ep_ref = (p_close_ref_x, p_close_ref_y, p_far_ref_x, p_far_ref_y)
#
#     libs.visualize.visualize_match_and_epipolar(stereo_1.key_frame.get_image(),
#                                               stereo_1.reference_frame.get_image(), (x, y), (match_x, match_y),
#                                               ep_ref,(epx,epy,rescale_factor,K_t_vector),root_dir+'epipolar/',str(idx))


# computation
#(successes, matches,matches_for_large_disparities_in_x,marked_keypoints) = depth_estimation_re.depth_estimation()

#visualisation
#libs.visualize.visualize_successes_and_discarded_pixels(stereo_pair_builder.key_frame.get_image(),marked_keypoints)
#libs.visualize.show(stereo_pair_builder.key_frame.get_depth_map(),normalize=True)