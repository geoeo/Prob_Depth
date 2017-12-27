import libs.vo_parser_KITTI as VO_PARSER
import libs.camera_model as CAMERA_MODEL
import libs.image_loader_KITTI as IMAGE_LOADER
import libs.stereo_pair_builder as STEREO_PAIR_BUILDER
import libs.visualize as VISUALIZE
import libs.settings as SETTINGS
import libs.runner as RUNNER

def bulk_run():

    log_file = open('datasets/kitti/dataset/log_file.txt','w+')
    img_range = range(361,800)

    for key_img in img_range:
        info = 'Key: ' + str(key_img) + '\n'
        print(info + '\n')
        log_file.write(info)
        ref_img = key_img + 1
        runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key_img, [ref_img], root_dir)
        runner.run_check_pixels(log_file,post_process=True,debug_mode=False,skip_guards=True)

    log_file.close()

def check_gradients():
    log_file = open('datasets/kitti/dataset/gradients.txt','w+')
    img_range = range(0, 1)
    for key_img in img_range:
        info = 'Key: ' + str(key_img) + ' '
        log_file.write(info)
        print(info + '\n')
        ref_img = key_img + 1
        runner = RUNNER.Runner(stereo_pair_builder, runtime_settings, key_img, [ref_img], root_dir)
        runner.check_gradient(log_file,post_process=True,debug_mode=False,skip_guards=True)


    log_file.close()

root_dir = 'datasets/kitti/dataset/'
sequence = '00'
images_dir = 'sequences/'+ sequence +'/'
# left_cam_dir = 'image_0/'
# left_cam_dir = 'image_0_resize_x/'
left_cam_dir = 'image_0_640_480/'
right_cam_dir = 'image_1/' # Unsued for now
odom_path = 'poses/'
# depth_file = root_dir + 'data/depth/img0001_0.depth'

# key = 0
key = 63
# key = 136
# key = 140
# key = 250
# key = 360
# key = 456
# key = 108
# key = 145
# key = 1152

# cm = CAMERA_MODEL.CameraModel(718.8560000000, 718.8560000000, 607.1928000000, 185.2157000000)
# cm = CAMERA_MODEL.CameraModel(718.8560000000*0.5157131346, 718.8560000000, 320, 185.2157000000)
cm = CAMERA_MODEL.CameraModel(718.8560000000*0.5157131346, 718.8560000000*1.2765957447, 320, 185.2157000000*1.2765957447) #640x480
image_loader = IMAGE_LOADER.ImageLoader(root_dir+images_dir+left_cam_dir)
vo = VO_PARSER.VoParserSynth(root_dir,sequence)
runtime_settings = SETTINGS.Settings(cm.focal_length[0],
                                     cm.focal_length[1],
                                     cm.principal_point[0],
                                     cm.principal_point[1],
                                     40, 0.0001, 0.8, 1.0,0.05)

ref_count = 2
ref_list = []
for count in range(1, ref_count + 1):
    ref_list.append(key + count)

stereo_pair_builder = STEREO_PAIR_BUILDER.StereoPairBuilder(cm, image_loader, vo, 0, runtime_settings)
runner = RUNNER.Runner(stereo_pair_builder,runtime_settings,key,ref_list,root_dir)

runner.run(VISUALIZE.visualize_enum.SHOW_DEPTH, normalize=True, regularize=True, show_frame=True, debug_mode=False, skip_guards = True)

# runner.run_show_epipolar_segments(100,skip_guards=True)
# runner.run_compare_gradients(200,skip_guards=True)
# bulk_run()

# check_gradients()






