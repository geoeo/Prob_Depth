import libs.vo_parser_ipad as VO_PARSER
import libs.camera_model as CAMERA_MODEL
import libs.image_loader_ipad as IMAGE_LOADER
import libs.stereo_pair_builder as STEREO_PAIR_BUILDER
import libs.visualize as VISUALIZE
import libs.settings as SETTINGS
import libs.runner as RUNNER

def bulk_run():
    log_file = open('datasets/ipad_small/log_file.txt', 'w+')
    img_range = range(361, 800)

    for key_img in img_range:
        info = 'Key: ' + str(key_img) + '\n'
        print(info + '\n')
        log_file.write(info)
        ref_img = key_img + 1
        runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key_img, [ref_img], root_dir)
        runner.run_check_pixels(log_file, post_process=True, debug_mode=False, skip_guards=True)

    log_file.close()


def check_gradients():
    log_file = open('datasets/ipad_small/gradients.txt', 'w+')
    img_range = range(0, 1)
    for key_img in img_range:
        info = 'Key: ' + str(key_img) + ' '
        log_file.write(info)
        print(info + '\n')
        ref_img = key_img + 1
        runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key_img, [ref_img], root_dir)
        runner.check_gradient(log_file, post_process=False, debug_mode=False, skip_guards=True)

    log_file.close()


root_dir = 'datasets/ipad_small/'
#dataset = 'home_640_480/'
dataset = 'home_512_424/'
# dataset = 'iviso_640_480/'
type ='x-translation/'
# type ='y-translation/'
#type ='z-translation/'
# type ='z-translation-5/'
# type ='y-translation-7/'
# type ='z-translation-4-couch/'
#type ='z-translation-4/'
# type ='z-translation-3/'
# type ='z-translation-10/'
# type =''
sequence = '2/'
images_dir = 'Images/'
# images_dir = 'Images_contrast/'
odom_path = 'Poses/'
intrinsics_dir = 'Intrinsics/'
mse_dir = root_dir+'mse/'
depth_file = root_dir + dataset+'Kinect_GT/y-trans-41/02-36-27/raw-ushort-512-424.txt'

# depth_folder = root_dir + 'data/depth/'
# depth_file = IMAGE_LOADER.ImageLoader.parse_id(key) + '.depth'
# depth_path = depth_folder + depth_file

# key = 0
# key = 17
# key = 28
key = 30 #** 512,x,3,0.75
# key = 41 #**512,y,6, 0.9
# key = 50
# key = 60 #** 512,z-8,4 step 5
# key = 70
# key = 70
# key = 71
# key = 75
# key = 80 #640,z,3,-1.0, 512, z ,4-couch,1
# key = 92 #* 512,z,3
# key = 100
# key = 101
# key = 110
# key = 120 #* z-trans_640 4,1
#key = 130 # z-trans_512, 4, 2
# key = 131
# key = 140
# key = 150
# key = 151 # # z-trans_512, 4, 2
# key = 160
# key = 162
# key = 170
# key = 180
# key = 190
# key = 181
# key = 200 #* z-trans
# key = 229
# key = 240 # z-trans_512 1
# key = 245 #* 512,1
# key = 250
# key = 260
# key = 290
# key = 320
# key = 340
# key = 400
# key = 480
# key = 482
# key = 910

cm = CAMERA_MODEL.CameraModel.load_from_file(root_dir + dataset + type + sequence + intrinsics_dir + 'intrinsics.txt', key)
image_loader = IMAGE_LOADER.ImageLoader(root_dir + dataset + type + sequence + images_dir)
vo = VO_PARSER.VoParserSynth(root_dir, dataset+ type + sequence + odom_path)
runtime_settings = SETTINGS.Settings(10, 0.0001, 0.9, 5, 0.05, -1.0) # -1.0 for z (or general motion eq.) TODO investigate

ref_count =8
step = 1
ref_list = []
for count in range(step, ref_count + 1, step):
    ref_list.append(key + count)
stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 0, runtime_settings)

runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key, ref_list, root_dir)

# ground_truth = POST.Evaluation.load_ground_truth(depth_file, 424, 512,flip_across_y=True)
# inverted_ground_truth = POST.Evaluation.calc_inverse_ground_truth(ground_truth, runtime_settings,max_thresh=2500,isIpad=True)
inverted_ground_truth = None

# VISUALIZE.show_frame(inverted_ground_truth, runtime_settings, path=root_dir, name='ground_truth', cmap='nipy_spectral')
runner.run(VISUALIZE.visualize_enum.SHOW_DEPTH, inverted_ground_truth, normalize=True, calc_error_metrics=False, post_process=True,
           regularize=True, show_frame=True, debug_mode=False, skip_guards=True)

# runner.run_show_epipolar_segments(50,skip_guards=True)

# runner.run_compare_gradients(200,skip_guards=True)
# bulk_run()

# check_gradients()



# f_avg = (runtime_settings.f_x + runtime_settings.f_y)/2.0
# idmap = stereo_pair_builder.key_frame.get_depth_map().astype(float)
# dmap = idmap.copy()
# #dmap[idmap>1.0] = f_avg/idmap[idmap>1.0]
# plt.imshow(idmap)
# plt.show()
# fx=stereo_pair_builder.return_fx(stereo_pair_builder.intrinsic)
# fy=runtime_settings.f_y
# cx=runtime_settings.c_x
# cy=runtime_settings.c_y
# PC=POINTCLOUD.PointCloud(np.array(getXYZ(dmap,fx,fy,cx,cy)))
# PC.show()
