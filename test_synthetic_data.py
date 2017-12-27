import libs.vo_parser_synth as VO_PARSER
import libs.camera_model as CAMERA_MODEL
import libs.image_loader_synth as IMAGE_LOADER
import libs.stereo_pair_builder as STEREO_PAIR_BUILDER
import depth_estimation as DEPTH_ESTIMATION
import libs.settings as SETTINGS
import libs.visualize as VISUALIZE
import libs.runner as RUNNER
import libs.PostProcess as POST
import numpy as np

root_dir = 'datasets/synth/'
images_dir = root_dir + 'data/img'
odom_path = root_dir + 'info/groundtruth.txt'
mse_dir = root_dir+'mse/'

key = 1

depth_folder = root_dir + 'data/depth/'
depth_file = IMAGE_LOADER.ImageLoader.parse_id(key) + '.depth'
depth_path = depth_folder + depth_file

cm = CAMERA_MODEL.CameraModel(329.115520046, 329.115520046, 320.0, 240.0)
image_loader = IMAGE_LOADER.ImageLoader(images_dir)
vo = VO_PARSER.VoParserSynth(odom_path)
runtime_settings = SETTINGS.Settings(20, 0.0001, 0.8, 1.0, 0.05, 1.0)

ref_count = 1
ref_list = []
for count in range(1, ref_count + 1):
    ref_list.append(key + count)

def load_and_plot_mse():
    mse_list = POST.Evaluation.load_data_list(mse_dir+str(key)+'/'+'mse_list.txt')
    name = str(key) + '_mse_line_plot'
    VISUALIZE.line_graph(mse_list,root_dir,name)

def load_and_bar_chart_mse():
    mse_list = POST.Evaluation.load_data_list(mse_dir + str(key) + '/' + 'mse_list.txt')
    min = np.amin(mse_list)
    # best avg values form middlebury evaluation page
    # http://vision.middlebury.edu/stereo/eval3/
    best_avg_sparse = 4.61
    best_avg_dense = 12.9
    data = [min,best_avg_sparse,best_avg_dense]
    name = str(key) + '_mse_bar_plot'
    VISUALIZE.bar_chart(data, root_dir, name)

# stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 0, runtime_settings)
# runner = RUNNER.Runner(stereo_pair_builder,runtime_settings,key,ref_list,root_dir)
# stereo_pairs = stereo_pair_builder.generate_stereo_pairs(key, ref_list, invert=False)

stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 0, runtime_settings)
runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key, ref_list, root_dir)

# ground_truth = POST.Evaluation.load_ground_truth(depth_path, 480, 640)
# inverted_ground_truth = POST.Evaluation.calc_inverse_ground_truth(ground_truth, runtime_settings, 329.115520046)
# VISUALIZE.show_frame(inverted_ground_truth, runtime_settings, path=root_dir, name='ground_truth', cmap='nipy_spectral')
#
# runner.run(VISUALIZE.visualize_enum.SHOW_DEPTH, inverted_ground_truth, normalize=True, calc_error_metrics=True,
#            post_process=True, regularize=True,
#            show_frame=True, skip_guards=False)

runner.run_show_epipolar_segments(200)

# loads mse values from text file; no need to run the calculation
load_and_plot_mse()
# load_and_bar_chart_mse()
