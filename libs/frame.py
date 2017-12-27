import numpy as np
import cv2
import libs.settings as SETTINGS
import csv


class Frame:
    def __init__(self, image, time_stamp):
        #self.image_color = image
        mode = image.mode
        if mode == 'L': # http://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
            self.image = np.array(image)
        else:
            self.image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY) #uint8 images
        self.image_float = np.float32(self.image)
        self.time_stamp = time_stamp
        (height, width) = self.image.shape  # assuming all images are same size
        self.height = height
        self.width = width
        self.depth_map = np.array(self.image, float)  # should be 1 channel image
        self.depth_map_buffer = np.array(self.image, float)  # should be 1 channel image
        self.variance_map = np.array(self.image, float)  # should be 1 channel image
        self.variance_map_buffer = np.array(self.image, float)  # should be 1 channel image
        self.validity_map = np.array(self.image, int)  # should be 1 channel image
        self.hypothesis_map = np.array(self.image, bool)
        self.absolute_gradient = None # absolute gradient of image
        self.init_hypothesis_randomly()
        self.img_grad_x = np.array(self.image, float)
        self.img_grad_y = np.array(self.image, float)
        self.precompute_gradients()

    """
        Initialize Depth Map and Variance Map Randomly
    """
    def init_hypothesis_randomly(self):
        self.depth_map = np.random.normal(0.5,0.00001,(self.height,self.width))
        self.variance_map = np.full((self.height,self.width),SETTINGS.Settings.var_random_init())
        self.hypothesis_map = np.full((self.height,self.width),False).astype(bool)

    def get_depth_map(self):
        return self.depth_map

    def get_variance_map(self):
        return self.variance_map

    def get_depth_map_smoothed(self):
        return self.depth_map_buffer

    def get_variance_map_smoothed(self):
        return self.variance_map_buffer

    def get_validity_map(self):
        return self.validity_map

    def get_image(self):
        return self.image

    def set_idepth(self,x,y,idepth):
        self.depth_map[y,x] = idepth

    def set_var(self, x, y, var):
        self.variance_map[y, x] = var

    def set_idepth_buffer(self,x,y,idepth):
        self.depth_map_buffer[y, x] = idepth

    def set_var_buffer(self, x, y, var):
        self.variance_map_buffer[y, x] = var

    def set_depth_hypothesis(self,x,y,has_hypothesis):
        self.hypothesis_map[y,x] = has_hypothesis

    def validity_inc(self,x,y,increment):
        self.validity_map[y,x] += increment

    def validity_dec(self,x,y,decrement):
        self.validity_map[y,x] -= decrement

    def get_depth_hypothesis(self,x,y):
        return self.hypothesis_map[y,x]

    def get_hypothesis_map(self):
        return self.hypothesis_map

    def get_img_grad_x_with(self,x,y):
        return self.img_grad_x[y,x]

    def get_img_grad_y_with(self,x,y):
        return self.img_grad_y[y,x]

    def get_img_grad_x(self):
        return self.img_grad_x

    def get_img_grad_y(self):
        return self.img_grad_y

    def precompute_gradients(self):
        # TODO:
        # ksize = 1 to resemble original implementation. use something higher if 
        # possible! also remove the astype(int)
        self.img_grad_x_float = cv2.Sobel(self.image,cv2.CV_32F,1,0,ksize=1)
        self.img_grad_y_float = cv2.Sobel(self.image,cv2.CV_32F,0,1,ksize=1)
        self.img_grad_x = self.img_grad_x_float.astype(int)
        self.img_grad_y = self.img_grad_y_float.astype(int)
        

    def init_depthmap_with_file(self, path,use_hypothesis=True):

        file = open(path, 'r')
        reader = csv.reader(file, delimiter=' ')
        data = []
        max_depth = 1/SETTINGS.Settings.min_depth()
        max_raw_depth = 0
        MAX = 80

        for row in reader:
            data.append(row)

        data = data[0]

        for x in range(0,self.width,1):
            for y in range(0,self.height,1):
                hypothesis_val =self.hypothesis_map[y,x]
                if use_hypothesis and hypothesis_val or not use_hypothesis:
                    if not use_hypothesis:
                        self.hypothesis_map[y, x] = True
                    idx = y*self.width + x
                    depth_raw = float(data[idx])
                    self.depth_map[y,x] = depth_raw
                    if depth_raw > max_raw_depth and depth_raw < MAX:
                        max_raw_depth = depth_raw

        for x in range(0, self.width, 1):
            for y in range(0, self.height, 1):
                hypothesis_val = self.hypothesis_map[y, x]
                if hypothesis_val:
                    depth_val = self.depth_map[y, x]
                    if depth_val >= MAX:
                        depth_val = SETTINGS.Settings.division_eps()
                    else:
                        depth_val /= MAX # normalize 0..1
                        depth_val = 1 - depth_val # invert
                        depth_val *= max_depth # scale
                    self.depth_map[y, x] = depth_val





    # bilinear interpolation between floor(x), floor(y), floor(x) + 1, floor(y) + 1
    # http://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    @staticmethod
    def bilinear_interp_(sample_map,x_in,y_in):

        x = np.asarray(x_in)
        y = np.asanyarray(y_in)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        # range check
        x0 = np.clip(x0, 0, sample_map.shape[1] - 1)
        x1 = np.clip(x1, 0, sample_map.shape[1] - 1)
        y0 = np.clip(y0, 0, sample_map.shape[0] - 1)
        y1 = np.clip(y1, 0, sample_map.shape[0] - 1)

        Ia = sample_map[y0,x0]
        Ib = sample_map[y1,x0]
        Ic = sample_map[y0, x1]
        Id = sample_map[y1,x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    @staticmethod
    def bilinear_interp(sample_map,x_in,y_in):
        return cv2.getRectSubPix(sample_map,(1,1),(x_in,y_in))[0][0]






