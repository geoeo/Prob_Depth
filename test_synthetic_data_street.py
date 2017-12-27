import libs.vo_parser_synth_street as VO_PARSER
import libs.camera_model as CAMERA_MODEL
import libs.image_loader_synth as IMAGE_LOADER
import libs.stereo_pair_builder as STEREO_PAIR_BUILDER
import libs.settings as SETTINGS
import libs.visualize as VISUALIZE
import libs.runner as RUNNER
import libs.PostProcess as POST

root_dir = 'datasets/synth_street/'
images_dir = root_dir + 'data/img'
mse_dir = root_dir+'mse/'
odom_path = root_dir + 'info/groundtruth.txt'

# key = 1 #*
# key = 10 #*
# key = 70
# key = 113
# key = 150 #*
# key = 324
# key = 325 #*
# key = 480
# key = 495 #*
# key = 500
# key = 536
# key = 583 # Rotation
key = 605 #*
# key = 1700



depth_folder = root_dir + 'data/depth/'
depth_file = IMAGE_LOADER.ImageLoader.parse_id(key) + '.depth'
depth_path = depth_folder + depth_file


def check_gradients():
    log_file = open('datasets/synth_street/gradients.txt', 'w+')
    img_range = range(10, 11)
    for key_img in img_range:
        info = 'Key: ' + str(key_img) + '\n'
        log_file.write(info)
        print(info + '\n')
        ref_img = key_img + 1
        runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key_img, [ref_img], root_dir)
        runner.check_gradient(log_file, post_process=True, debug_mode=False, skip_guards=False)

    log_file.close()

def load_and_plot_mse():
    mse_list = POST.Evaluation.load_data_list(mse_dir+str(key)+'/'+'mse_list.txt')
    name = str(key) + '_mse_line_plot'
    VISUALIZE.line_graph(mse_list,root_dir,name)

def load_and_bar_chart_avg_mse():
    avg_min = POST.Evaluation.calc_avg_mse(root_dir)
    # best avg values form middlebury evaluation page
    # http://vision.middlebury.edu/stereo/eval3/
    best_avg_sparse = 4.61
    best_avg_dense = 12.9
    data = [avg_min,best_avg_sparse,best_avg_dense]
    name = 'avg_mse_bar_plot'
    VISUALIZE.bar_chart(data, root_dir, name)




cm = CAMERA_MODEL.CameraModel(329.115520046, 329.115520046, 320.0, 240.0)
image_loader = IMAGE_LOADER.ImageLoader(images_dir)
vo = VO_PARSER.VoParserSynth(odom_path)
runtime_settings = SETTINGS.Settings(10, 0.0001, 0.8, 10.0, 0.05, 1.0)  # TODO ADD RESCALE FACTOR HERE

ref_count = 10
ref_list = []
for count in range(1, ref_count + 1):
    ref_list.append(key + count)

stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 0, runtime_settings)
runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key, ref_list, root_dir)

ground_truth = POST.Evaluation.load_ground_truth(depth_path, 480, 640)
inverted_ground_truth = POST.Evaluation.calc_inverse_ground_truth(ground_truth, runtime_settings, 329.115520046)
# VISUALIZE.show_frame(inverted_ground_truth, runtime_settings, path=root_dir, name='ground_truth', cmap='nipy_spectral')
runner.run(VISUALIZE.visualize_enum.SHOW_DEPTH, inverted_ground_truth, normalize=True, calc_error_metrics=False,
        post_process=True, regularize=True,
           show_frame=True, skip_guards=False)
# loads mse values from text file; no need to run the calculation
# load_and_plot_mse()
#
# runner.run_show_epipolar_segments(200)
# runner.run_compare_gradients(200)

# check_gradients()

# loads mse values from text file; no need to run the calculation
# load_and_bar_chart_avg_mse()
