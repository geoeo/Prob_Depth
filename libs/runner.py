import depth_estimation as DEPTH_ESTIMATION
import libs.visualize as VISUALIZE
import libs.PostProcess as POST
import cv2
import numpy as np


class Runner:
    def __init__(self, stereo_pair_builder, settings, keyframe_id, reference_frame_list, root_dir):
        self.root_dir = root_dir
        self.stereo_pair_builder = stereo_pair_builder
        self.settings = settings
        self.keyframe_id = keyframe_id
        self.reference_frame_list = reference_frame_list
        self.stereo_pairs = stereo_pair_builder.generate_stereo_pairs(keyframe_id, reference_frame_list, invert=False)
        self.mse_list = []

    def run(self, visualize_enum, ground_truth, normalize=True, calc_error_metrics = False,post_process = False, regularize=True, show_frame=True, debug_mode=False,
            skip_guards=False):
        iteration = 1
        stereo_pair_builder = self.stereo_pair_builder
        runtime_settings = stereo_pair_builder.runtime_settings

        for stereo_pair in self.stereo_pairs:
            it_string = str(iteration)
            early_exit = False
            if visualize_enum == VISUALIZE.visualize_enum.SHOW_INITIAL_GOOD_PIXELS:
                early_exit = True

            depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, runtime_settings)
            (good_pixels, matches, matches_for_large_disparities_in_x,
             marked_keypoints) = depth_estimation_re.depth_estimation(
                post_processing_enabled=regularize, early_exit=early_exit, debug_mode=debug_mode,
                skip_guards=skip_guards)

            hypothesis_map  = stereo_pair_builder.key_frame.get_hypothesis_map()

            post_processed_depth = stereo_pair_builder.key_frame.get_depth_map()

            if post_process:
                # post_processed_depth = POST.Evaluation.threshold(post_processed_depth, 1e-2,
                #                                                  5)
                post_processed_depth = POST.Evaluation.scale_matrix(post_processed_depth, runtime_settings.min_depth,
                                                                    runtime_settings.max_depth,
                                                                    hypothesis_map)


            if calc_error_metrics and not ground_truth is None:
                mse = POST.Evaluation.compute_rms(ground_truth, post_processed_depth,
                                                  hypothesis_map)
                self.mse_list.append(mse)
                print(mse)


            if visualize_enum == VISUALIZE.visualize_enum.SHOW_DEPTH:

                VISUALIZE.show(stereo_pair_builder.key_frame.get_image(),
                               post_processed_depth,
                               hypothesis_map,
                               stereo_pair_builder.runtime_settings,
                               normalize=normalize,
                               iteration=it_string,
                               path=self.root_dir, color_map='nipy_spectral', show_keyframe=show_frame)

            elif visualize_enum == VISUALIZE.visualize_enum.SHOW_VARIANCE:
                VISUALIZE.show(stereo_pair_builder.key_frame.get_image(),
                               stereo_pair_builder.key_frame.get_variance_map(),
                               hypothesis_map,
                               stereo_pair_builder.runtime_settings,
                               normalize=normalize,
                               iteration=it_string,
                               path=self.root_dir, color_map='magma')

            elif visualize_enum == VISUALIZE.visualize_enum.SHOW_SUCCESSES_AND_DISCARDED_PIXELS:
                VISUALIZE.visualize_successes_and_discarded_pixels(stereo_pair_builder.key_frame.get_image(),
                                                                   marked_keypoints,
                                                                   self.root_dir, iteration)
            elif visualize_enum == VISUALIZE.visualize_enum.SHOW_INITIAL_GOOD_PIXELS:
                VISUALIZE.visualize_keypoints(stereo_pair_builder.key_frame.get_image(), good_pixels, self.root_dir)

            iteration = iteration + 1

        if calc_error_metrics:
            # VISUALIZE.line_graph(self.mse_list, self.root_dir, 'line_graph')
            self.save_list(self.mse_list)

    def run_show_epipolar_segments(self, segment_step_size, skip_guards=False):
        stereo_pair = self.stereo_pairs[0]
        depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, self.stereo_pair_builder.runtime_settings)
        (successes, matches, matches_for_large_disparities_in_x,
         marked_keypoints) = depth_estimation_re.depth_estimation(post_processing_enabled=False, early_exit=False,
                                                                  skip_guards=skip_guards)
        match_amount = len(matches)

        disp_image_1 = stereo_pair.key_frame.get_image()
        disp_image_2 = stereo_pair.reference_frame.get_image()
        disp_image_1 = cv2.cvtColor(disp_image_1, cv2.COLOR_GRAY2BGR)
        disp_image_2 = cv2.cvtColor(disp_image_2, cv2.COLOR_GRAY2BGR)

        for idx in range(0, match_amount, segment_step_size):
            (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref, rescale_factor) = matches[idx]

            p_close_ref_x = p_close_ref[0, 0]
            p_close_ref_y = p_close_ref[1, 0]
            p_far_ref_x = p_far_ref[0, 0]
            p_far_ref_y = p_far_ref[1, 0]

            ep_ref = (p_close_ref_x, p_close_ref_y, p_far_ref_x, p_far_ref_y)

            VISUALIZE.visualize_match_and_epipolar(disp_image_1, disp_image_2, (x, y),
                                                       (match_x, match_y),
                                                       ep_ref, (epx, epy, rescale_factor),
                                                       self.root_dir + 'epipolar/', str(idx))
        res = np.zeros((disp_image_1.shape[0], disp_image_1.shape[1] * 2, disp_image_1.shape[2]))
        res[:, 0:disp_image_1.shape[1], :] = disp_image_1
        res[:, disp_image_1.shape[1]:, :] = disp_image_2

        cv2.imwrite(self.root_dir + 'epipolar/' + 'epipoloar.png', res)

    def save_list(self,data):
        file  = open(self.root_dir+'mse/'+str(self.keyframe_id)+'/'+'mse_list.txt', 'w')
        for data in data:
            file.write("%s\n" % str(data))
        print("saved data to %s" % self.root_dir)
        file.close()


    def run_compare_gradients(self, segment_step_size, skip_guards=False):
        stereo_pair = self.stereo_pairs[0]
        depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, self.stereo_pair_builder.runtime_settings)
        (successes, matches, matches_for_large_disparities_in_x,
         marked_keypoints) = depth_estimation_re.depth_estimation(post_processing_enabled=False, early_exit=False,
                                                                  skip_guards=skip_guards)
        match_amount = len(matches)

        for idx in range(0, match_amount, segment_step_size):
            (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref, rescale_factor) = matches[idx]

            p_close_ref_x = p_close_ref[0, 0]
            p_close_ref_y = p_close_ref[1, 0]
            p_far_ref_x = p_far_ref[0, 0]
            p_far_ref_y = p_far_ref[1, 0]

            ep_ref = (p_close_ref_x, p_close_ref_y, p_far_ref_x, p_far_ref_y)

            VISUALIZE.visualize_compare_gradients(stereo_pair.key_frame.get_image(),
                                                  stereo_pair.reference_frame.get_image(), (x, y),
                                                  (match_x, match_y),
                                                  ep_ref, (epx, epy, rescale_factor),
                                                  self.root_dir + 'grad_comp/', str(idx))

    def run_check_pixels(self, log_file, post_process=True, debug_mode=False, skip_guards=False):

        if log_file is None:
            return

        stereo_pair = self.stereo_pairs[0]
        stereo_pair_builder = self.stereo_pair_builder

        depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, stereo_pair_builder.runtime_settings)
        (good_pixels, matches, matches_for_large_disparities_in_x,
         marked_keypoints) = depth_estimation_re.depth_estimation(
            post_processing_enabled=post_process, early_exit=False, debug_mode=debug_mode, skip_guards=skip_guards)

        match_count = 0
        error_count = 0

        for (x, y, error_code) in marked_keypoints:
            if error_code == 0 or error_code == 10 or error_code == 20:
                match_count += 1
            else:
                error_count += 1

        out = "Success: " + str(match_count) + " - Discarded: " + str(error_count) + "\n"

        log_file.write(out)
        print(out)

    def check_gradient(self, log_file, post_process=True, debug_mode=False, skip_guards=False):

        if log_file is None:
            return

        stereo_pair = self.stereo_pairs[0]
        stereo_pair_builder = self.stereo_pair_builder

        depth_estimation_re = DEPTH_ESTIMATION.DepthEstimation(stereo_pair, stereo_pair_builder.runtime_settings)
        (good_pixels, matches, matches_for_large_disparities_in_x,
         marked_keypoints) = depth_estimation_re.depth_estimation(
            post_processing_enabled=post_process, early_exit=False, debug_mode=debug_mode, skip_guards=skip_guards)
        match_amount = len(matches)

        for idx in range(0, match_amount, 1):
            (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref, rescale_factor) = matches[idx]

            p_close_ref_x = p_close_ref[0, 0]
            p_close_ref_y = p_close_ref[1, 0]
            p_far_ref_x = p_far_ref[0, 0]
            p_far_ref_y = p_far_ref[1, 0]

            ep_ref = (p_close_ref_x, p_close_ref_y, p_far_ref_x, p_far_ref_y)
            inc_x = p_close_ref[0, 0] - p_far_ref[0, 0]
            inc_y = p_close_ref[1, 0] - p_far_ref[1, 0]

            key_ratio_y = epy / epx
            ref_ratio_y = inc_y / inc_x

            info = 'Gradient Ratio (x to) in Key Frame: 1:' + str(key_ratio_y) + ' in reference frame: 1:' + str(
                ref_ratio_y) + '\n'
            log_file.write(info)
            print(info)
