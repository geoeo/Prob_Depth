import libs.frame as FRAME
import libs.settings as SETTINGS
import math
import numpy as np


# reference implemntation https://github.com/geoeo/lsd_slam/blob/master/lsd_slam_core/src/DepthEstimation/DepthMap.cpp
# function : doLineStereo

class DepthEstimation:
    def __init__(self, current_stereo_pair, runtime_settings):
        self.stereo_pair = current_stereo_pair
        self.key_frame = current_stereo_pair.key_frame  # copy references to make de-referencing faster
        self.reference_frame = current_stereo_pair.reference_frame  # copy references to make de-referencing faster
        self.height = current_stereo_pair.reference_frame.height
        self.width = current_stereo_pair.reference_frame.width
        self.runtime_settings = runtime_settings
        # Debug data constructs
        self.rescale_factor_list = []
        self.match_error_list = []
        self.debug_mode = False
        self.skip_guards = False

    def depth_estimation(self, post_processing_enabled=True, early_exit=False, debug_mode = False, skip_guards = False):

        self.debug_mode = debug_mode
        self.skip_guards = skip_guards

        # Search for good candidate pixels
        (successes, good_pixels) = self.search_for_good_pixels()

        matches = []
        matches_for_large_disparities_in_x = []
        keypoints = []
        err_in_search_space_counter = 0

        if early_exit:
            return (good_pixels, matches, matches_for_large_disparities_in_x, keypoints)

        for (x, y, epx, epy) in good_pixels:
            if not self.key_frame.get_depth_hypothesis(x, y):
                (success, match, p_close_ref, p_far_ref, marked_keypoints, rescale_factor) = self.create_depth_estimate(
                    x, y, epx, epy)
            else:
                (success, match, p_close_ref, p_far_ref, marked_keypoints, rescale_factor) = self.update_depth_estimate(
                    x, y, epx, epy)

            (x, y, code) = marked_keypoints
            if success:
                (x, y, match_x, match_y) = match
                if code == 20:
                    matches.append(
                        (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref, rescale_factor))
                elif code == 10:
                    matches.append(
                        (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref, rescale_factor))
                    matches_for_large_disparities_in_x.append(
                        (x, y, match_x, match_y, epx, epy, p_close_ref, p_far_ref))
            if code == 1:
                err_in_search_space_counter += 1
            keypoints.append(marked_keypoints)

        # Post Processing
        if post_processing_enabled:
            self.denoise()
            self.depth_map_smoothing()

        if self.debug_mode:
            DepthEstimation.print_list_stats("Rescale Factor", self.rescale_factor_list)
            DepthEstimation.print_list_stats("Best Match Error", self.match_error_list)

        return (good_pixels, matches, matches_for_large_disparities_in_x, keypoints)

    def search_for_good_pixels_(self):

        successes = 0
        success = False
        good_pixels = []

        for y in range(3, self.height - 3, 1):
            for x in range(3, self.width - 3, 1):

                g_x = self.key_frame.img_grad_x[y, x]
                g_y = self.key_frame.img_grad_y[y, x]
                absolute_gradient = math.sqrt(g_x * g_x + g_y * g_y)

                if self.key_frame.get_depth_hypothesis(x,y) and absolute_gradient < self.runtime_settings.min_abs_grad_decrease():
                    self.key_frame.set_depth_hypothesis(x, y, False)
                    continue
                if absolute_gradient < self.runtime_settings.min_abs_grad_create():
                    continue

                (success, epx, epy) = self.make_and_check_epl(x, y)

                if success:
                    successes = successes + 1
                    good_pixels.append((x, y, epx, epy))

        return successes, good_pixels

    def search_for_good_pixels(self):
        successes = 0
        success = False
        good_pixels = []

        absolute_gradient = np.sqrt(np.square(self.key_frame.img_grad_x_float) + np.square(self.key_frame.img_grad_y_float))
        
        depth_hypothesis = self.key_frame.hypothesis_map
        
        X_,Y_ = np.meshgrid(range(self.width),range(self.height))
        cond = (((absolute_gradient < self.runtime_settings.min_abs_grad_decrease()) & 
                self.key_frame.hypothesis_map) | 
                (absolute_gradient < self.runtime_settings.min_abs_grad_create()))
        
        X_valid = X_[~cond]
        Y_valid = Y_[~cond]
                
        self.key_frame.hypothesis_map[absolute_gradient < self.runtime_settings.min_abs_grad_decrease()] = False
        
        (success_, epx_, epy_,x_,y_) = self.make_and_check_epl(X_valid, Y_valid)

        if not success_.any():
            return 0 , []
        
        for success, epx, epy,x,y in zip(success_, epx_, epy_,x_,y_):
          successes = successes + 1
          good_pixels.append((x, y, epx, epy))
        # print(len(good_pixels), self.height*self.width, successes)
        return successes, good_pixels


    def create_depth_estimate(self, x, y, epx, epy):

        # Init values since no prior hypothesis exists
        min_idepth = 0
        max_idepth = 1 / self.runtime_settings.get_min_depth()
        prior_idepth = 1.0

        (success, result_idepth, result_var, result_epl_length, x_match, y_match, p_close_ref,
         p_far_ref, rescale_factor, marked_keypoints) = self.compute_inverse_depth(x, y, epx, epy, min_idepth,
                                                                                   prior_idepth, max_idepth)

        if not success:
            return (False, (0, 0, 0, 0), 0, 0, marked_keypoints, rescale_factor)

        result_idepth = SETTINGS.Settings.UNZERO(result_idepth)

        self.key_frame.set_idepth(x, y, result_idepth)
        self.key_frame.set_var(x, y, result_var)
        self.key_frame.set_depth_hypothesis(x, y, True)

        return (success, (x, y, x_match, y_match), p_close_ref, p_far_ref, marked_keypoints, rescale_factor)

    def update_depth_estimate(self, x, y, epx, epy):

        prior_var = self.key_frame.get_variance_map()[y, x]
        sv = math.sqrt(prior_var)
        prior_idepth = self.key_frame.get_depth_map()[y, x]
        min_idepth = prior_idepth - sv * self.runtime_settings.stereo_epl_var_fac()
        max_idepth = prior_idepth + sv * self.runtime_settings.stereo_epl_var_fac()

        max = 1 / self.runtime_settings.get_min_depth()

        if (min_idepth < 0): min_idepth = 0
        if max_idepth > max: max_idepth = max

        (success, result_idepth, result_var, result_epl_length, x_match, y_match, p_close_ref,
         p_far_ref, rescale_factor, marked_keypoints) = self.compute_inverse_depth(x, y, epx, epy, min_idepth,
                                                                                   prior_idepth, max_idepth)

        if not success:
            return (False, (0, 0, 0, 0), 0, 0, marked_keypoints, rescale_factor)

        # EKF UPDATE

        # increase var by a little (prediction-uncertainty)
        id_var = prior_var * SETTINGS.Settings.succ_var_inc_fac()

        # update var with observation
        w = result_var / (result_var + id_var)
        new_idepth = (1 - w) * result_idepth + w * prior_idepth

        new_idepth = SETTINGS.Settings.UNZERO(new_idepth)

        # variance can only decrease from observation; never increase.
        id_var *= w
        if id_var < prior_var:
            self.key_frame.set_var(x, y, id_var)

        self.key_frame.set_idepth(x, y, new_idepth)

        return (success, (x, y, x_match, y_match), p_close_ref, p_far_ref, marked_keypoints, rescale_factor)

    def make_and_check_epl_(self, x, y):
        t_x = self.stereo_pair.ref_to_key[0, 3]
        t_y = self.stereo_pair.ref_to_key[1, 3]
        t_z = self.stereo_pair.ref_to_key[2, 3]
        # normalize intrinsics so that standard values can be used
        f_x = self.stereo_pair.return_fx(self.stereo_pair.intrinsic)
        f_y = self.stereo_pair.return_fy(self.stereo_pair.intrinsic)
        c_x = self.stereo_pair.return_cx(self.stereo_pair.intrinsic)
        c_y = self.stereo_pair.return_cy(self.stereo_pair.intrinsic)

        # skew symmteric fundametal matrix
        F = np.zeros(shape=(3, 3))

        F[0, 0] = 0
        F[1, 0] = t_z
        F[2, 0] = -f_y * t_y - c_y * t_z
        F[0, 1] = -t_z
        F[1, 1] = 0
        F[2, 1] = f_x * t_x + c_x * t_z
        F[0, 2] = f_y * t_y + c_y * t_z
        F[1, 2] = -f_x * t_x - c_x * t_z
        F[2, 2] = 0

        uv = np.array([[x], [y], [1]])

        res = np.matmul(F, uv)

        # flip normal so that we obtain the gradient
        epx_from_fundamental = res[1, 0] / self.width
        epy_from_fundamental = -res[0, 0] / self.height

        # u = x/self.width
        # v = y/self.height

        # TODO investigate space of this point
        # epx = -f_x * t_x + t_z * (x - c_x)
        # epy = -f_y * t_y + t_z * (y - c_y)
        #
        # epx /= self.width
        # epy /= self.height

        epx = epx_from_fundamental
        epy = epy_from_fundamental

        K_t_vector = (f_x * t_x + c_x * t_z, f_y * t_y + c_y * t_z, t_z)

        if math.isnan(epx + epy):
            return (False, 0, 0)

        # check epl length
        epl_length_squared = epx * epx + epy * epy
        # if epl_length_squared < self.runtime_settings.min_epl_length_squared():
        if epl_length_squared == 0.0:
            return (False, 0, 0)

        # check epl-grad magnitude
        gx = self.key_frame.img_grad_x[y,x]
        gy = self.key_frame.img_grad_y[y,x]
        epl_grad_squared = gx * epx + gy * epy
        epl_sq = epl_grad_squared * epl_grad_squared
        epl_grad_squared = epl_sq / epl_length_squared  # square and normalize

        if epl_grad_squared < self.runtime_settings.min_epl_grad_squared():
            return (False, 0, 0)

        # check epl angle - important for geometric disparity assumption
        angle = epl_grad_squared / (gx * gx + gy * gy)

        # my angle check using dot product
        e_norm = math.sqrt(epx * epx + epy * epy)
        g_norm = math.sqrt(gx * gx + gy * gy)

        # angle = ((epx / e_norm) * (gx / g_norm)) + ((epy / e_norm) * (gy / g_norm))

        angle_thresh = self.runtime_settings.min_epl_angle_squared()
        if (angle < angle_thresh):
            return (False, 0, 0)

        fac = self.runtime_settings.gradient_sample_dist() / math.sqrt(epl_length_squared)

        epx_norm = epx * fac
        epy_norm = epy * fac

        return (True, epx_norm, epy_norm)

    def make_and_check_epl(self, x, y):
        t_x = self.stereo_pair.ref_to_key[0, 3]
        t_y = self.stereo_pair.ref_to_key[1, 3]
        t_z = self.stereo_pair.ref_to_key[2, 3]
        # normalize intrinsics so that standard values can be used
        f_x = self.stereo_pair.return_fx(self.stereo_pair.intrinsic)
        f_y = self.stereo_pair.return_fy(self.stereo_pair.intrinsic)
        c_x = self.stereo_pair.return_cx(self.stereo_pair.intrinsic)
        c_y = self.stereo_pair.return_cy(self.stereo_pair.intrinsic)

        # skew symmteric fundametal matrix
        F = np.zeros(shape=(3, 3))

        F[0, 0] = 0
        F[1, 0] = t_z
        F[2, 0] = -f_y * t_y - c_y * t_z
        F[0, 1] = -t_z
        F[1, 1] = 0
        F[2, 1] = f_x * t_x + c_x * t_z
        F[0, 2] = f_y * t_y + c_y * t_z
        F[1, 2] = -f_x * t_x - c_x * t_z
        F[2, 2] = 0

        # no translation
        if not F.any():
            return (np.full((1, 1), False), 0, 0, 0, 0)

        uv = np.vstack((x,y,np.ones_like(x)))

        res = np.matrix(F)*np.matrix(uv)

        epx_from_fundamental = res[1, :] / self.width
        epy_from_fundamental = -res[0, :] / self.height
        
        #result vectors
        success = np.full_like(x,False).astype(bool)
        epx_norm = np.zeros_like(x)
        epy_norm = np.zeros_like(x)
        
        
        epx = np.array(epx_from_fundamental[0,:].flat)
        epy = np.array(epy_from_fundamental[0,:].flat)

        # check epl length
        ## does not need to be done as these are default anyway!
        epl_length_squared = np.square(epx) + np.square(epy)

        # check epl-grad magnitude
        gx = self.key_frame.img_grad_x[y,x].astype(float) #this can also be done if x and y are arrays
        gy = self.key_frame.img_grad_y[y,x].astype(float) #this can also be done if x and y are arrays
        epl_grad_squared = np.array(gx) * np.array(epx) + np.array(gy) * np.array(epy)
        epl_sq = np.square(epl_grad_squared)
        epl_grad_squared = epl_sq / epl_length_squared  # square and normalize

        # check epl angle - important for geometric disparity assumption
        sq_abs_grad_sq = np.square(gx) + np.square(gy)
        angle = epl_grad_squared / sq_abs_grad_sq

        # my angle check using dot product
        e_norm = np.sqrt(epl_length_squared)
        g_norm = np.sqrt(sq_abs_grad_sq)

        ## does not need to be done as these are default anyway!
        angle_thresh = self.runtime_settings.min_epl_angle_squared()

        fac = self.runtime_settings.gradient_sample_dist() / e_norm

        # build idx for successful calculations


        idx = np.where((angle >= angle_thresh) &
               (epl_grad_squared >= self.runtime_settings.min_epl_grad_squared()) &
               (epl_length_squared>=1e-6) &
               ~np.isnan(epx + epy))[0]

        epx_norm = np.array(epx[idx]) * np.array(fac[idx])
        epy_norm = np.array(epy[idx]) * np.array(fac[idx])
        success[idx] = True

        return (success[idx], epx_norm, epy_norm, x[idx], y[idx])

    def compute_inverse_depth(self, u_key, v_key, epxn, epyn, min_idepth, prior_idepth, max_idepth):

        current_stereo_pair = self.stereo_pair
        error_in_calc_search_space = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_key, v_key, 1))
        error_in_matching = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_key, v_key, 2))
        error_in_depth_computation = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_key, v_key, 3))

        # normalized image coordiantes
        p = np.array([[u_key], [v_key], [1]])
        k_inv = current_stereo_pair.intrinsic.K_inv
        k_inv_p = np.matmul(k_inv, p)

        (success, p_close, p_far, inc_x, inc_y, rescale_factor, epl_length,
         flip_epl_point_to_other_side) = self.calc_search_space_in_reference_frame(
            k_inv_p,
            u_key,
            v_key, epxn,
            epyn,
            min_idepth,
            prior_idepth,
            max_idepth)

        if not success:
            return error_in_calc_search_space

        result_epl_length = epl_length

        key_frame_image = current_stereo_pair.key_frame.image_float

        # values we want to search for in our reference frame
        real_val_p1 = FRAME.Frame.bilinear_interp(key_frame_image,
                                                  u_key + epxn * rescale_factor * flip_epl_point_to_other_side,
                                                  v_key + epyn * rescale_factor * flip_epl_point_to_other_side)
                
        real_val_m1 = FRAME.Frame.bilinear_interp(key_frame_image,
                                                  u_key - epxn * rescale_factor * flip_epl_point_to_other_side,
                                                  v_key - epyn * rescale_factor * flip_epl_point_to_other_side)
        real_val = FRAME.Frame.bilinear_interp(key_frame_image, u_key, v_key)
        real_val_m2 = FRAME.Frame.bilinear_interp(key_frame_image,
                                                  u_key - 2 * epxn * rescale_factor * flip_epl_point_to_other_side,
                                                  v_key - 2 * epyn * rescale_factor * flip_epl_point_to_other_side)
        real_val_p2 = FRAME.Frame.bilinear_interp(key_frame_image,
                                                  u_key + 2 * epxn * rescale_factor * flip_epl_point_to_other_side,
                                                  v_key + 2 * epyn * rescale_factor * flip_epl_point_to_other_side)

        # we start at p_far, and inc is in direction of (p_far to p_close)
        # slightly unintuitive

        cp_x = p_far[0, 0]
        cp_y = p_far[1, 0]

        # first 5 points in the reference frame

        ref_frame_image = current_stereo_pair.reference_frame.image_float

        val_cp_m2 = FRAME.Frame.bilinear_interp(ref_frame_image, cp_x - 2.0 * inc_x, cp_y - 2.0 * inc_y)
        val_cp_m1 = FRAME.Frame.bilinear_interp(ref_frame_image, cp_x - inc_x, cp_y - inc_y)
        val_cp = FRAME.Frame.bilinear_interp(ref_frame_image, cp_x, cp_y)
        val_cp_p1 = FRAME.Frame.bilinear_interp(ref_frame_image, cp_x + inc_x, cp_y + inc_y)
        # val_cp_p2 - declaration here for consistency

        best_match_x_val, best_match_y_val, sample_dist, grad_along_line, success = self.match_stereo(
            cp_x, cp_y, current_stereo_pair, inc_x, inc_y, p_close, real_val, real_val_m1, real_val_m2, real_val_p1,
            real_val_p2, val_cp, val_cp_m1, val_cp_m2, val_cp_p1, rescale_factor)

        if not success:
            return error_in_matching

        (hasNewDepth, area_of_match, inverse_depth_new, alpha) = self.calc_depth_in_key_frame(inc_x, inc_y,
                                                                                              best_match_x_val,
                                                                                              best_match_y_val,
                                                                                              k_inv_p, u_key, v_key)

        if not hasNewDepth:
            return error_in_depth_computation

        var_new = self.calc_new_variance(alpha, u_key, v_key, epxn, epyn, sample_dist, grad_along_line)

        return (True, inverse_depth_new, var_new, result_epl_length, best_match_x_val, best_match_y_val, p_close, p_far,
                rescale_factor, (u_key, v_key, area_of_match))

    def match_stereo(self, cp_x, cp_y, current_stereo_pair, inc_x, inc_y, p_close, real_val, real_val_m1, real_val_m2,
                     real_val_p1, real_val_p2, val_cp, val_cp_m1, val_cp_m2, val_cp_p1, rescale_factor):

        success = True
        loop_counter = 0
        best_match_x = -1
        best_match_y = -1
        best_match_err = 1e50
        second_best_match_err = 1e50
        best_was_last_loop = False
        loop_c_best = -1
        loop_c_second = -1
        error_total_last = -1
        ref_frame_image = current_stereo_pair.reference_frame.get_image()

        best_match_errPre = 1e50
        best_match_errPost = 1e50
        best_match_DiffErrPre = 1e50
        best_match_DiffErrPost = 1e50

        e1A = 0
        e2A = 0
        e3A = 0
        e4A = 0
        e5A = 0

        e1B = 0
        e2B = 0
        e3B = 0
        e4B = 0
        e5B = 0

        ee_last = -1

        # while we are starting or we can still go along the epipolar line
        while (((inc_x < 0) == (cp_x > p_close[0, 0]) and (inc_y < 0) == (cp_y > p_close[1, 0])) or loop_counter == 0):
            val_cp_p2 = FRAME.Frame.bilinear_interp(ref_frame_image, cp_x + 2.0 * inc_x, cp_y + 2.0 * inc_y)
            error_total = 0

            if loop_counter % 2 == 0:

                e1A = val_cp_p2 - real_val_p2
                e2A = val_cp_p1 - real_val_p1
                e3A = val_cp - real_val
                e4A = val_cp_m1 - real_val_m1
                e5A = val_cp_m2 - real_val_m2

                error_total = e1A * e1A + e2A * e2A + e3A * e3A + e4A * e4A + e5A * e5A

            else:
                e1B = val_cp_p2 - real_val_p2
                e2B = val_cp_p1 - real_val_p1
                e3B = val_cp - real_val
                e4B = val_cp_m1 - real_val_m1
                e5B = val_cp_m2 - real_val_m2

                error_total = e1B * e1B + e2B * e2B + e3B * e3B + e4B * e4B + e5B * e5B

            if (error_total < best_match_err):

                second_best_match_err = best_match_err
                loop_c_second = loop_c_best

                best_match_err = error_total
                loop_c_best = loop_counter

                if self.debug_mode:
                    self.match_error_list.append(float(best_match_err))

                best_match_x = cp_x
                best_match_y = cp_y
                best_was_last_loop = True

                best_match_errPre = ee_last
                best_match_DiffErrPre = e1A * e1B + e2A * e2B + e3A * e3B + e4A * e4B + e5A * e5B
                best_match_errPost = -1
                best_match_DiffErrPost = -1

            else:
                if error_total < second_best_match_err:
                    second_best_match_err = error_total
                    loop_c_second = loop_counter
                    best_was_last_loop = False

                if (error_total < second_best_match_err):
                    second_best_match_err = error_total
                    loop_c_second = loop_counter

            error_total_last = error_total
            # shift every measurement on the epipolar line one step further

            ee_last = error_total

            val_cp_m2 = val_cp_m1
            val_cp_m1 = val_cp
            val_cp = val_cp_p1
            val_cp_p1 = val_cp_p2

            cp_x += inc_x
            cp_y += inc_y

            loop_counter += 1

        # if the best match error is still too big
        if not self. skip_guards and (best_match_err > 4.0 * self.runtime_settings.max_error_stereo()):
            success = False

        if not self.skip_guards and (abs(loop_c_best - loop_c_second) > 1 and self.runtime_settings.min_distance_error_stereo() * best_match_err > second_best_match_err and success):
            success = False

        (best_match_x, best_match_y, best_match_err) = self.sub_pixel_interpolation(best_match_errPre, best_match_DiffErrPre,
                                                                               best_match_err, best_match_DiffErrPost,
                                                                               best_match_errPost, best_match_x,
                                                                               best_match_y, inc_x, inc_y)

        # sample_dist is the distance in pixel at which the realVal's were sampled
        sample_dist = self.runtime_settings.gradient_sample_dist() * rescale_factor
        grad_along_line = 0

        tmp = real_val_p2 - real_val_p1
        grad_along_line += tmp * tmp

        tmp = real_val_p1 - real_val
        grad_along_line += tmp * tmp

        tmp = real_val - real_val_m1
        grad_along_line += tmp * tmp

        tmp = real_val_m1 - real_val_m2
        grad_along_line += tmp * tmp

        grad_along_line /= sample_dist * sample_dist

        # check if interpolated error is OK.use evil hack to allow more error if there is
        # a lot of gradient.
        thresh = self.runtime_settings.max_error_stereo() + math.sqrt(grad_along_line) * 20.0
        if not self.skip_guards and (float(best_match_err) > thresh and success):
            success = False

        return best_match_x, best_match_y, sample_dist, grad_along_line, success

    # *KinvP = Kinv * (x, y, 1); where x, y are pixel coordinates
    #  of point we search for , in the key frame.
    # *best_match_x = x - coordinate of found correspondence in the reference frame.
    def calc_depth_in_key_frame(self, inc_x, inc_y, best_match_x, best_match_y, k_inv_p, u, v):
        current_stereo_pair = self.stereo_pair

        match_in_reference = np.array([[best_match_x], [best_match_y], [1]])
        key_point = np.array([[u], [v], [1]])
        k_inv = current_stereo_pair.intrinsic_ref.K_inv
        match_in_reference_inv_p = np.matmul(k_inv, match_in_reference)
        key_to_ref_translation = current_stereo_pair.return_key_to_ref_translation()
        key_to_ref_rotation = current_stereo_pair.return_key_to_ref_rotation()
        f_xi = current_stereo_pair.return_fxi(current_stereo_pair.intrinsic_ref)
        f_yi = current_stereo_pair.return_fyi(current_stereo_pair.intrinsic_ref)
        f_x = current_stereo_pair.return_fx(current_stereo_pair.intrinsic_ref)
        f_y = current_stereo_pair.return_fy(current_stereo_pair.intrinsic_ref)
        c_x = current_stereo_pair.return_cx(current_stereo_pair.intrinsic_ref)
        c_y = current_stereo_pair.return_fy(current_stereo_pair.intrinsic_ref)
        success = True
        area_of_match = 0  # x = 10, y = 20

        z_trans = key_to_ref_translation[2, 0]
        y_trans = key_to_ref_translation[1, 0]
        x_trans = key_to_ref_translation[
            0, 0]

        if (inc_x * inc_x > inc_y * inc_y):
            area_of_match = 10
            match_in_ref_inv_x = match_in_reference_inv_p[0, 0]
            z_trans = key_to_ref_translation[2, 0]
            x_trans = key_to_ref_translation[
                0, 0]
            baseline = x_trans- match_in_ref_inv_x * z_trans
            # baseline = abs(x_trans- match_in_ref_inv_x * z_trans)
            # baseline = math.sqrt(x_trans * x_trans + z_trans * z_trans*match_in_ref_inv_x*match_in_ref_inv_x)
            # baseline = match_in_ref_inv_x * z_trans + x_trans
            # baseline = match_in_ref_inv_x * z_trans - x_trans # left side for z, with 1.0
            baseline_2 = -match_in_ref_inv_x * z_trans - x_trans
            # baseline = math.fabs(z_trans)


            row_0 = key_to_ref_rotation[0, :]
            row_2 = key_to_ref_rotation[2, :]
            dot_0 = np.dot(row_0, k_inv_p)
            dot_0 = dot_0[0]
            dot_2 = np.dot(row_2, k_inv_p)[0]
            x_r = match_in_ref_inv_x * dot_2

            x_d = dot_0 - x_r
            # x_d = abs(abs(dot_0) - abs(x_r))
            # x_d = abs(dot_0) - abs(x_r)
            # x_d_2 = abs(abs(x_r) - abs(dot_0))

            # x_d = dot_0 + match_in_ref_inv_x * dot_2
            # x_d =  match_in_ref_inv_x * dot_2 - dot_0 # ipad - z
            # x_d_2 =  -match_in_ref_inv_x * dot_2 - dot_0
            # x_d *=-1
            inverse_depth_new = x_d / baseline

            # INVERSE DEPTH VIA GENERAL MOTION EQ FOR CAMERA AS IN MULTIPLE VIEW GEOMETRY
            denom = f_x*x_trans + c_x*z_trans
            numerator = best_match_x -c_x*dot_2 - f_x*dot_0

            inverse_depth_new = numerator / denom

            # if inverse_depth_new < 0:
            #     inverse_depth_new = x_d_2 / baseline_2
            inner_1 = (dot_0 * z_trans - dot_2 * x_trans)
            alpha = inc_x * f_xi * inner_1 / (
                baseline * baseline)
        else:
            # return (False, area_of_match, -1, -1)
            area_of_match = 20
            match_in_ref_inv_y = match_in_reference_inv_p[1, 0]
            z_trans = key_to_ref_translation[2, 0]
            y_trans = key_to_ref_translation[
                1, 0]
            baseline =  y_trans - match_in_ref_inv_y * z_trans
            # baseline =  abs(y_trans - match_in_ref_inv_y * z_trans)
            # baseline =  math.sqrt(y_trans*y_trans + z_trans*z_trans*match_in_ref_inv_y*match_in_ref_inv_y)
            # baseline =  match_in_ref_inv_y * z_trans + y_trans
            # baseline =  match_in_ref_inv_y * z_trans - y_trans # left side for z, with 1.0
            baseline_2 =  -match_in_ref_inv_y * z_trans - y_trans
            # baseline = math.fabs(z_trans)


            row_1 = key_to_ref_rotation[1, :]
            row_2 = key_to_ref_rotation[2, :]
            dot_1 = np.dot(row_1, k_inv_p)[0]
            dot_2 = np.dot(row_2, k_inv_p)[0]
            x_r = match_in_ref_inv_y * dot_2
            y_d = dot_1 - x_r
            # y_d = abs(abs(dot_1) - abs(x_r))
            # y_d = abs(dot_1) - abs(x_r)
            # y_d_2 = abs(abs(x_r) - abs(dot_1))


            # y_d = dot_1 + match_in_ref_inv_y * dot_2
            # y_d_2 = match_in_ref_inv_y * dot_2 - dot_1 # ipad- z
            y_d_2 = -match_in_ref_inv_y * dot_2 - dot_1
            inverse_depth_new = y_d / baseline

            # INVERSE DEPTH VIA GENERAL MOTION EQ FOR CAMERA AS IN MULTIPLE VIEW GEOMETRY
            denom = f_y*y_trans + c_y*z_trans
            numerator = best_match_y + c_y*dot_2 + f_y*dot_1

            inverse_depth_new = numerator / denom
            # if inverse_depth_new < 0:
            #     inverse_depth_new = y_d_2 / baseline_2
            inner_1 = (dot_1 * z_trans - dot_2 * y_trans)
            alpha = inc_y * f_yi * inner_1 / (
                baseline * baseline)
        inverse_depth_new *= self.runtime_settings.invert_depth # set for street_data off for ipad - TODO investigate this
        # inverse_depth_new = math.fabs(inverse_depth_new)
        if inverse_depth_new == 0.0:
            inverse_depth_new = self.runtime_settings.get_min_depth()
        elif inverse_depth_new < 0:
            return (False, area_of_match, -1, -1)

        # alpha - ratio of length of searched inverse depth interval to length of the searched epipolar line
        return (success, area_of_match, inverse_depth_new, alpha)

    def calc_new_variance(self, alpha, u, v, epxn, epyn, sample_dist, grad_along_line):

        current_stereo_pair = self.stereo_pair

        photo_disp_error = 4.0 * self.runtime_settings.camera_pixel_noise() / (
            grad_along_line + self.runtime_settings.division_eps())
        tracking_error = 0.25  # Todo investigate residual error
        grad_x = FRAME.Frame.bilinear_interp(current_stereo_pair.key_frame.img_grad_x_float, u, v)
        grad_y = FRAME.Frame.bilinear_interp(current_stereo_pair.key_frame.img_grad_y_float, u, v)
        geo_disp_error = (grad_x * epxn + grad_y * epyn) + self.runtime_settings.division_eps()
        geo_disp_error = tracking_error * tracking_error * (
            grad_x * grad_x + grad_y * grad_y) / (geo_disp_error * geo_disp_error)
        sub_pixel_term = 0.5  # Todo adjust when subpixel is done
        # sub_pixel_term = 0.05 # when sub pixel is active
        result_variance = alpha * alpha * sub_pixel_term * sample_dist * sample_dist + geo_disp_error + photo_disp_error
        return result_variance

    # Computes the start and end point in 3D space, and the gradient of the 2D line connecting them
    # returns tuple ( success (bool), p_close, p_far, incx, incy, rescale_factor)
    def calc_search_space_in_reference_frame(self, k_inv_p, u_key, v_key, epxn, epyn, min_idepth, prior_idepth,
                                             max_idepth):

        current_stereo_pair = self.stereo_pair
        error = (False, 0, 0, 0, 0, 0, 0, 0)

        # align point with reference frame camera
        p_inf = np.matmul(current_stereo_pair.return_k_key_to_ref_rotation(), k_inv_p)
        k_key_to_ref_translation = current_stereo_pair.return_k_key_to_ref_translation()

        key_to_ref_translation = current_stereo_pair.return_key_to_ref_translation()

        key_to_ref_x = key_to_ref_translation[0, 0]
        key_to_ref_y = key_to_ref_translation[1, 0]
        key_to_ref_z = key_to_ref_translation[2, 0]
        is_z_translation = False
        if (key_to_ref_z * key_to_ref_z > key_to_ref_x * key_to_ref_x and
                        key_to_ref_z * key_to_ref_z > key_to_ref_y * key_to_ref_y):
            is_z_translation = True

        # Flag dictates if the close point gets flipped to the other side
        flip_epl_point_to_other_side = 1.0
        # Have systematic error when translating in x-space. p_close is on the wrong side
        if not is_z_translation:
            flip_epl_point_to_other_side = -1.0

        # our estimate of the point in the reference frame coordiante system
        p_real = p_inf + k_key_to_ref_translation * prior_idepth

        rescale_factor = p_real[2, 0]

        if self.debug_mode:
            self.rescale_factor_list.append(rescale_factor)

        # TODO where do these numbers come from?
        # 0.7 - 1.4 Default
        #  1.9 for KITTI
        if not self.skip_guards and (rescale_factor <= 0.7 or rescale_factor >= 1.4):
            return error

        trans_max = k_key_to_ref_translation * max_idepth * flip_epl_point_to_other_side
        p_close_real = p_inf + trans_max

        # # if the assumed close point lies behind the image plane we change that
        # if p_close_real[2, 0] < 0.001:
        #     max_idepth = (0.001 - p_inf[2, 0]) / k_key_to_ref_translation[2]
        #     p_close_real = p_inf + k_key_to_ref_translation * max_idepth * invert_epl

        # position of point in reference frame assuming max_idepth
        p_close = p_close_real / p_close_real[2, 0]

        trans_min = k_key_to_ref_translation * min_idepth
        p_far_real = p_inf + trans_min

        # if the assumed far-point lies behind the image or closer than the near-point,
        # we moved past the Point it and should stop.
        if p_close[2, 0] < 0.001 or max_idepth < min_idepth:
            return error

        p_far = p_far_real / p_far_real[2, 0]

        if math.isnan(p_far[0] + p_close[0].astype(float)):
            return (False, 0, 0, 0, 0, 0, 0, 0)

        # calculate increments in which we will step through the epipolar line

        inc_x = p_close[0, 0] - p_far[0, 0]
        inc_y = p_close[1, 0] - p_far[1, 0]

        epl_length = math.sqrt(inc_x * inc_x + inc_y * inc_y)
        if epl_length <= 0 or math.isinf(epl_length):
            return error

        if epl_length > self.runtime_settings.max_epl_length_crop():
            p_close[0, 0] = p_far[0, 0] + inc_x * self.runtime_settings.max_epl_length_crop() / epl_length
            p_close[1, 0] = p_far[1, 0] + inc_y * self.runtime_settings.max_epl_length_crop() / epl_length

        inc_x *= self.runtime_settings.gradient_sample_dist() / epl_length
        inc_y *= self.runtime_settings.gradient_sample_dist() / epl_length
        scale = self.runtime_settings.get_epl_scale()

        #extend one sample_dist to left & right.
        # p_far[0] -= inc_x
        # p_far[1] -= inc_y
        # p_close[0] += inc_x
        # p_close[1] += inc_y

        # # This switches the position of the epipolar end point to the other side in screen space
        # Have systematic error when translating in x-space. p_close is on the wrong side
        # if (inc_x*inc_x > inc_y*inc_y and is_z_translation):
        # if (p_close_real[2] > p_far_real[2] and is_z_translation):
        #     p_close[0, 0] = p_far[0, 0] - inc_x  * rescale_factor * max_idepth
            # p_close[1, 0] = p_far[1, 0] - inc_y  * rescale_factor * max_idepth
            # inc_x = -inc_x
            # inc_y = -inc_y
            # flip_epl_point_to_other_side = -1.0

        # fist time extend search range
        # TODO investigate this
        # extend oe sample dist to left & right
        if scale != 0.0:
            p_far[0, 0] -= scale * inc_x
            p_far[1, 0] -= scale * inc_y
            p_close[0, 0] += 1.0 * inc_x
            p_close[1, 0] += 1.0 * inc_y

        # if epl_length < self.runtime_settings.min_epl_length_crop():
        #     pad = (self.runtime_settings.min_epl_length_crop() - epl_length) / 2
        #     p_far[0, 0] -= inc_x * pad
        #     p_far[1, 0] -= inc_y * pad
        #
        #     p_close[0, 0] += inc_x * pad
        #     p_close[1, 0] += inc_y * pad

        sample_point_to_border = self.runtime_settings.sample_point_to_border()
        # sample_point_to_border = 1.0

        # if far point is outside of image, cant be back projected -> skip
        if (p_far[0, 0] <= sample_point_to_border or
                    p_far[0, 0] >= self.width - sample_point_to_border or
                    p_far[1, 0] <= sample_point_to_border or
                    p_far[1, 0] >= self.height - sample_point_to_border
            ):
            return error

        # if near point is outside of image, cant be back projected -> skip
        # TODO refernece implementaiton adjusts this and tries again
        if (p_close[0, 0] <= sample_point_to_border or
                    p_close[0, 0] >= self.width - sample_point_to_border or
                    p_close[1, 0] <= sample_point_to_border or
                    p_close[1, 0] >= self.height - sample_point_to_border
            ):
            return error


        return (True, p_close, p_far, inc_x, inc_y, rescale_factor, epl_length, flip_epl_point_to_other_side)

    def sub_pixel_interpolation(self, best_match_errPre, best_match_DiffErrPre, best_match_err, best_match_DiffErrPost,
                                best_match_errPost, best_match_x, best_match_y, inc_x, inc_y):

        gradPre_pre = -(best_match_errPre - best_match_DiffErrPre)
        gradPre_this = +(best_match_err - best_match_DiffErrPre)
        gradPost_this = -(best_match_err - best_match_DiffErrPost)
        gradPost_post = +(best_match_errPost - best_match_DiffErrPost)

        interp_post = False
        interp_pre = False

        if (gradPre_pre < 0) ^ (gradPre_this < 0):

            # if post has zero - crossing
            if not ((gradPost_post < 0) ^ (gradPost_this < 0)):
                interp_pre = True

        # if post has zero-crossing
        elif ((gradPost_post < 0) ^ (gradPost_this < 0)):
            interp_post = True

        if (interp_pre):
            d = gradPre_this / (gradPre_this - gradPre_pre)
            best_match_x -= d * inc_x
            best_match_y -= d * inc_y
            best_match_err = best_match_err - 2 * d * gradPre_this - (gradPre_pre - gradPre_this) * d * d



        elif interp_post:
            d = gradPost_this / (gradPost_this - gradPost_post)
            best_match_x += d * inc_x
            best_match_y += d * inc_y
            best_match_err = best_match_err + 2 * d * gradPost_this + (gradPost_post - gradPost_this) * d * d

        return (best_match_x, best_match_y, best_match_err)

    def denoise(self):
        depth_map = self.stereo_pair.key_frame.get_depth_map()
        variance_map = self.stereo_pair.key_frame.get_variance_map()
        hypothesis_map = self.stereo_pair.key_frame.get_hypothesis_map()

        reg_radius = 2

        hypothesis_resets = []

        for y in range(reg_radius, self.height - reg_radius, 1):
            for x in range(reg_radius, self.width - reg_radius, 1):

                val_sum = 0

                src_depth = depth_map[y, x]
                src_var = variance_map[y, x]
                is_valid = hypothesis_map[y, x]

                if not is_valid:
                    continue

                for dy in range(-reg_radius, reg_radius + 1, 1):
                    for dx in range(-reg_radius, reg_radius + 1, 1):

                        # skip center
                        if dx == 0 and dy == 0:
                            continue

                        y_dest = y + dy
                        x_dest = x + dx

                        dest_depth = depth_map[y_dest, x_dest]
                        dest_var = variance_map[y_dest, x_dest]
                        dest_std = math.sqrt(dest_var)
                        dest_is_valid = hypothesis_map[y_dest, x_dest]

                        if (
                                dest_is_valid and src_depth <= dest_depth + 2 * dest_std and src_depth >= dest_depth - 2 * dest_std):
                            val_sum += 1

                # if hypothesis is not surrounded by enough other valid hyp. -> remove
                if val_sum < SETTINGS.Settings.val_sum_min_for_keep():
                    hypothesis_resets.append((x, y))

        for (x, y) in hypothesis_resets:
            hypothesis_map[y, x] = False
            depth_map[y, x] = np.random.normal(0.5, 0.00001)
            variance_map[y, x] = SETTINGS.Settings.var_random_init()

    # TODO for multi-threading we need 2 buffers
    def depth_map_smoothing(self):
        depth_map = self.stereo_pair.key_frame.get_depth_map()
        variance_map = self.stereo_pair.key_frame.get_variance_map()
        hypothesis_map = self.stereo_pair.key_frame.get_hypothesis_map()
        reg_radius = 2

        smoothed_values = []

        for y in range(reg_radius, self.height - reg_radius, 1):
            for x in range(reg_radius, self.width - reg_radius, 1):

                sum = 0
                val_sum = 0
                sumIvar = 0
                count = 0
                regDistVar = SETTINGS.Settings.reg_dist_var()

                src_count = 0
                src_sum = 0
                src_sumIvar = 0

                src_depth = depth_map[y, x]
                src_var = variance_map[y, x]
                is_valid = hypothesis_map[y, x]

                for dy in range(-reg_radius, reg_radius + 1, 1):
                    for dx in range(-reg_radius, reg_radius + 1, 1):

                        y_dest = y + dy
                        x_dest = x + dx

                        dest_is_valid = hypothesis_map[y_dest, x_dest]

                        if not dest_is_valid:
                            continue

                        dest_depth = depth_map[y_dest, x_dest]
                        dest_var = variance_map[y_dest, x_dest]

                        if is_valid:
                            count += 1
                            distFac = (dx * dx + dy * dy) * regDistVar
                            ivar = 1.0 / (dest_var + distFac)

                            sum += dest_depth * ivar
                            sumIvar += ivar

                        elif not is_valid:
                            src_count += 1
                            distFac = (dx * dx + dy * dy) * regDistVar
                            ivar = 1.0 / (dest_var + distFac)

                            src_sum += dest_depth * ivar
                            src_sumIvar += ivar

                if is_valid:
                    sum_avg = sum / sumIvar
                    # sum_avg =  sum / count
                    sum_avg = SETTINGS.Settings.UNZERO(sum_avg)
                    sumIvar /= count
                    smoothed_values.append((x, y, sum_avg, sumIvar))
                elif src_sumIvar != 0 and src_count > SETTINGS.Settings.min_val_for_fill():
                    sum_avg = src_sum / src_sumIvar
                    # sum_avg =  sum / count
                    sum_avg = SETTINGS.Settings.UNZERO(sum_avg)
                    src_sumIvar /= src_count
                    smoothed_values.append((x, y, sum_avg, src_sumIvar))

        for (x, y, sum, sumIvar) in smoothed_values:
            depth_map[y, x] = sum
            variance_map[y, x] = 1.0 / sumIvar
            hypothesis_map[y, x] = True


    @staticmethod
    def print_list_stats(header,list):

        if len(list) == 0:
            return

        print("STATS FOR: " + header )

        max_val = max(list)
        min_val = min(list)
        mean = np.mean(list)
        median = np.median(list)
        std = np.std(list)

        print("max val: " + str(max_val))
        print("min val: " + str(min_val))
        print("mean val: " + str(mean))
        print("median val: " + str(median))
        print("std val: " + str(std))

        print("STATS END")
