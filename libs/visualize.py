
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import matplotlib.colors
from enum import Enum
import libs.settings
import cv2

class visualize_enum(Enum):
    SHOW_DEPTH = 1
    SHOW_VARIANCE = 2
    SHOW_SUCCESSES_AND_DISCARDED_PIXELS = 3
    SHOW_INITIAL_GOOD_PIXELS = 4

def show_frame(grey_image,runtime_settings, path='', name='ground_truth',cmap='gray'):
    disp_image = grey_image.copy()
    (height, width) = disp_image.shape
    min_depth = runtime_settings.get_min_depth()
    max_depth = 1 / min_depth

    max = np.amax(disp_image)
    min = np.amin(disp_image)

    # plt.imshow(disp_image, cmap=cmap,norm=matplotlib.colors.LogNorm())
    plt.imshow(disp_image, cmap=cmap)
    # plt.imshow(disp_image, cmap=cmap,vmin=min_depth,vmax=max_depth)
    # plt.imshow(disp_image, cmap=cmap,vmin=0.01,vmax=max_depth,norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    # plt.show()
    plt.savefig(path + name)
    plt.close()

def show(keyframe,grey_image, hypothesis_map, runtime_settings,normalize=False, iteration='1', path='', color_map='gray',show_keyframe = False):
    disp_image = grey_image.copy()
    disp_image_ma = np.ma.array(disp_image, mask=~hypothesis_map)
    (height, width) = disp_image.shape
    min_depth = runtime_settings.get_min_depth()
    max_depth = 1 / min_depth

    # median = np.median(disp_image_ma)
    # mean = np.mean(disp_image_ma)
    max = np.amax(disp_image_ma)
    min = np.amin(disp_image_ma)

    #  if no depth values exist
    if not hypothesis_map.max():
        return

    if show_keyframe:
        plt.imshow(keyframe, cmap='gray')
    else:
        plt.imshow(hypothesis_map, cmap='gray')

    # plt.imshow(disp_image_ma, cmap=color_map, norm=matplotlib.colors.LogNorm())
    plt.imshow(disp_image_ma, cmap=color_map)
    # plt.imshow(disp_image_ma, cmap=color_map,vmin=0.01,vmax=0.1, norm=matplotlib.colors.LogNorm())
    # plt.imshow(disp_image_ma, cmap=color_map,vmin=0,vmax=vis_range)

    plt.colorbar()
    plt.savefig(path + 'iteration' + '_' + color_map + iteration)
    plt.close()


def show_hypothesis(hypothesis_map, iteration='1', path=''):
    (height, width) = hypothesis_map.shape
    disp_image = np.empty((height, width), float)

    for y in range(0, height, 1):
        for x in range(0, width, 1):
            if hypothesis_map[y, x]:
                disp_image[y, x] = 255
            else:
                disp_image[y, x] = 0

    plt.imshow(disp_image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.savefig(path + 'hypothesis_iteration' + '_' + iteration)
    plt.close()

def line_graph(data,path,name):
    X = [x+1 for x in range(0,len(data))]
    plt.plot(X,data,'bo')
    plt.xlabel('Number of Reference Frames Used')
    plt.ylabel('Root Mean Squared Error')
    plt.title('RMSE for Consecutive Iterations of the Depth Estimation')
    plt.savefig(path + name)
    plt.close()

def bar_chart(data,path,name):
    X = [x + 1 for x in range(0, len(data))]
    objects = ('Avg. Semi-Dense','Avg Sparse','Avg. Dense')
    plt.bar(X,data,width=0.5)
    plt.xticks(X, objects)
    plt.ylabel('Avg. Root Mean Squared Error')
    plt.title('Comparison of Avgerage Root Mean Squared Error Values')

    plt.savefig(path + name)
    plt.close()


def show_prob_custom(grey_image, disp_min, disp_max, path):
    (height, width) = grey_image.shape
    disp_image = grey_image.copy()
    disp_range = disp_max - disp_min

    for y in range(0, height, 1):
        for x in range(0, width, 1):
            v = disp_image[y, x] - disp_min
            v = v / disp_range
            disp_image[y, x] = v * 255

    plt.imshow(disp_image, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.show()


def visualize_keypoints(grey_image, points, path, circle_size=0.05):
    (height, width) = grey_image.shape
    disp_image = grey_image.copy()

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    title = "Initial Good Pixels in Key Frame: "
    amount =len(points)
    percentage = amount/(height*width)
    title += str(percentage)
    ax.set_title(title)

    patches = []

    for x, y, epx, epy in points:
        patches.append(plt.Circle((x, y), circle_size))

    p = PatchCollection(patches, facecolors='none', edgecolors='b')
    ax.add_collection(p)
    ax.imshow(disp_image, cmap='gray', vmin=0, vmax=255)
    plt.savefig(path + 'keypoints')

def visualize_matches(grey_image_key, grey_image_ref, p_key, p_ref, circle_size=1.0):
    disp_image_1 = grey_image_key.copy()
    disp_image_2 = grey_image_ref.copy()

    # (223,29) and (237.28,29.02)

    circle_key = plt.Circle(p_key, circle_size, facecolor='none', edgecolor='b')
    circle_ref = plt.Circle(p_ref, circle_size, facecolor='none', edgecolor='b')
    f, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.set_aspect('equal')
    ax1.imshow(disp_image_1, cmap='gray', vmin=0, vmax=255)
    ax1.add_patch(circle_key)
    ax1.set_title("Key Frame")

    # ax2.set_aspect('equal')
    ax2.imshow(disp_image_2, cmap='gray', vmin=0, vmax=255)
    ax2.add_patch(circle_ref)
    ax2.set_title("Reference Frame")

    # f,ax = plt.subplots(1)
    # ax.set_aspect('equal')
    # ax.imshow(disp_image_1, cmap='gray', vmin=0, vmax=255)
    # ax.add_patch(circle_key)

    plt.show()


def visualize_epipolar_lines(grey_image_key, grey_image_ref, p_key, p_ref):
    disp_image_1 = grey_image_key.copy()
    disp_image_2 = grey_image_ref.copy()
    (height, width) = disp_image_1.shape

    (x_key, y_key, epx_key, epy_key) = p_key

    (x_close_ref, y_close_ref, x_far_ref, y_far_ref) = p_ref

    ep_mag = math.sqrt(epx_key * epx_key + epy_key * epy_key)

    epx_key /= ep_mag
    epy_key /= ep_mag

    delta = min(width - x_key, height - y_key)

    x_key_far = x_key + delta * epx_key
    y_key_far = y_key + delta * epy_key

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(disp_image_1, cmap='gray', vmin=0, vmax=255)
    ax1.plot([x_key, x_key_far], [y_key, y_key_far], 'r-', linewidth=0.5)
    ax1.set_title("Key Frame")

    ax2.imshow(disp_image_2, cmap='gray', vmin=0, vmax=255)
    ax2.plot([x_close_ref, x_far_ref], [y_close_ref, y_far_ref], 'r-', linewidth=0.5)
    ax2.set_title("Reference Frame")

    plt.show()

def visualize_compare_gradients(grey_image_key, grey_image_ref, point_key, point_ref, ep_ref, key_epx, path, id,
                                 circle_size=1.0):
    disp_image_1 = grey_image_key.copy()
    # disp_image_2 = grey_image_ref.copy()

    (x_close_ref, y_close_ref, x_far_ref, y_far_ref) = ep_ref
    (epx, epy, rescale_factor) = key_epx
    (u_key, v_key) = point_key

    (x_close_key, y_close_key, x_far_key, y_far_key) = (
    u_key - 2 * epx * rescale_factor, v_key - 2 * epy * rescale_factor, u_key + 2 * epx * rescale_factor,
    v_key + 2 * epy * rescale_factor)

    circle_key = plt.Circle(point_key, circle_size, facecolor='none', edgecolor='b')
    circle_ref = plt.Circle(point_ref, circle_size, facecolor='none', edgecolor='b')
    circle_close_ref = plt.Circle((x_close_ref, y_close_ref,), circle_size / 2, facecolor='none', edgecolor='g')
    circle_far_ref = plt.Circle((x_far_ref, y_far_ref,), circle_size / 2, facecolor='none', edgecolor='r')

    fig, ax = plt.subplots()

    ax.imshow(disp_image_1, cmap='gray', vmin=0, vmax=255)
    # ax1.add_patch(circle_key)
    ax.plot([x_close_key, x_far_key], [y_close_key, y_far_key], 'y-', linewidth=0.5)
    ax.plot([x_close_ref, x_far_ref], [y_close_ref, y_far_ref], 'b-', linewidth=0.1)
    ax.set_title("Key Frame")

    # ax2.imshow(disp_image_2, cmap='gray', vmin=0, vmax=255)
    # ax2.add_patch(circle_ref)
    # ax2.add_patch(circle_close_ref)
    # ax2.add_patch(circle_far_ref)
    # ax2.plot([x_close_ref, x_far_ref], [y_close_ref, y_far_ref], 'y-', linewidth=0.5)
    # ax2.set_title("Reference Frame")

    plt.savefig(path + 'gradient_compare_' + id + '.svg', format='svg', dpi=300)
    plt.close()

def visualize_match_and_epipolar(grey_image_key, grey_image_ref, point_key, point_ref, ep_ref, key_epx, path, id,
                                 circle_size=1.0):
    disp_image_1 = grey_image_key.copy()
    disp_image_2 = grey_image_ref.copy()

    (x_close_ref, y_close_ref, x_far_ref, y_far_ref) = ep_ref
    (epx, epy, rescale_factor) = key_epx
    (u_key, v_key) = point_key

    (x_close_key, y_close_key, x_far_key, y_far_key) = (
    u_key - 2 * epx * rescale_factor, v_key - 2 * epy * rescale_factor, u_key + 2 * epx * rescale_factor,
    v_key + 2 * epy * rescale_factor)

    circle_key = plt.Circle(point_key, circle_size, facecolor='none', edgecolor='b')
    circle_ref = plt.Circle(point_ref, circle_size, facecolor='none', edgecolor='b')
    circle_close_ref = plt.Circle((x_close_ref, y_close_ref,), circle_size / 2, facecolor='none', edgecolor='g')
    circle_far_ref = plt.Circle((x_far_ref, y_far_ref,), circle_size / 2, facecolor='none', edgecolor='r')

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(disp_image_1, cmap='gray', vmin=0, vmax=255)
    ax1.add_patch(circle_key)
    ax1.plot([x_close_key, x_far_key], [y_close_key, y_far_key], 'y-', linewidth=0.5)
    ax1.set_title("Key Frame")

    ax2.imshow(disp_image_2, cmap='gray', vmin=0, vmax=255)
    ax2.add_patch(circle_ref)
    ax2.add_patch(circle_close_ref)
    ax2.add_patch(circle_far_ref)
    ax2.plot([x_close_ref, x_far_ref], [y_close_ref, y_far_ref], 'y-', linewidth=0.5)
    ax2.set_title("Reference Frame")

    plt.savefig(path + 'epipoloar_' + id + '.svg', format='svg', dpi=300)
    plt.close()

def visualize_match_and_epipolar_cv2(disp_image_1,disp_image_2,grey_image_key, grey_image_ref, point_key, point_ref, ep_ref, key_epx, path, id, circle_size=1):
    #disp_image_1 = grey_image_key.copy()
    #disp_image_2 = grey_image_ref.copy()

    (x_close_ref, y_close_ref, x_far_ref, y_far_ref) = ep_ref
    (epx, epy, rescale_factor) = key_epx
    (u_key, v_key) = point_key

    (x_close_key, y_close_key, x_far_key, y_far_key) = (
    u_key - 2 * epx * rescale_factor, v_key - 2 * epy * rescale_factor, u_key + 2 * epx * rescale_factor, v_key + 2 * epy * rescale_factor)

    #cv2.circle(img,center, radius, color, lthickness)
    disp_image_1 = cv2.circle(disp_image_1,(int(point_key[0]),int(point_key[1])), circle_size, (255,0,0), 1)
    disp_image_1 = cv2.line(disp_image_1,(int(x_close_key), int(y_close_key)), (int(x_far_key), int(y_far_key)),(0,255,255),1)
    
    disp_image_2 = cv2.circle(disp_image_2,(int(point_ref[0]),int(point_ref[1])), circle_size, (255,0,0), 1)
    disp_image_2 = cv2.circle(disp_image_2,(int(x_close_ref), int(y_close_ref)), circle_size/2, (0,255,0), 1)
    disp_image_2 = cv2.circle(disp_image_2,(int(x_far_ref), int(y_far_ref)), circle_size/2, (0,0,255), 1)
    disp_image_2 = cv2.line(disp_image_2,(int(x_close_ref), int(y_close_ref)), (int(x_far_ref), int(y_far_ref)),(0,255,255),1)
    
    


def visualize_successes_and_discarded_pixels(grey_image_key, marked_keypoints, path, iteration, circle_size=0.05):
    disp_image = grey_image_key.copy()

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.set_title("Keypoints in Key Frame")

    successes = []
    points_discarded_in_search_space_computation = []
    points_discarded_during_matching = []
    points_discarded_during_depth_computation = []

    for (x, y, error_code) in marked_keypoints:
        if error_code == 0 or error_code == 10 or error_code == 20:
            successes.append(plt.Circle((x, y), circle_size))
        elif error_code == 1:
            points_discarded_in_search_space_computation.append(plt.Circle((x, y), circle_size))
        elif error_code == 2:
            points_discarded_during_matching.append(plt.Circle((x, y), circle_size))
        elif error_code == 3:
            points_discarded_during_depth_computation.append(plt.Circle((x, y), circle_size))

    p = PatchCollection(successes, facecolors='none', edgecolors='b')
    p_2 = PatchCollection(points_discarded_in_search_space_computation, facecolors='none', edgecolors='r')
    p_3 = PatchCollection(points_discarded_during_matching, facecolors='none', edgecolors='g')
    p_4 = PatchCollection(points_discarded_during_depth_computation, facecolors='none', edgecolors='y')

    ax.add_collection(p)
    ax.add_collection(p_2)
    ax.add_collection(p_3)
    ax.add_collection(p_4)

    red_patch = mpatches.Patch(color='red', label='Too Close to Border')
    green_patch = mpatches.Patch(color='green', label='Failed During Matching')
    yellow_patch = mpatches.Patch(color='yellow', label='Failed Depth Calculation')
    blue_patch = mpatches.Patch(color='blue', label='Successful')
    plt.legend(handles=[red_patch,green_patch,yellow_patch,blue_patch])

    ax.imshow(disp_image, cmap='gray', vmin=0, vmax=255)
    plt.savefig(path + 'discarded' + '_' + str(iteration))
    plt.close()


