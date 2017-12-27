import csv
import numpy as np



class CameraModel:
    def __init__(self, f_x, f_y,c_x,c_y):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
        self.intrinsics = []

    @staticmethod
    def load_from_file(file_path,key_id):
        file = open(file_path, 'r')
        reader = csv.reader(file, delimiter=' ')
        reader_list = list(reader)

        row = reader_list[key_id]
        fx = float(row[0])
        fy= float(row[4])
        cx = float(row[6])
        cy = float(row[7])
        cm = CameraModel(fx,fy,cx,cy)
        cm.intrinsics = reader_list
        return cm

    @staticmethod
    def build_K_matrix_with_params(f_x,f_y,c_x,c_y):
        K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])

        return K


    def build_K_matrix(self):
        K = np.array([[self.f_x,0,self.c_x],
                      [0,self.f_y,self.c_y],
                      [0,0,1]])

        return K



    def build_K_matrix_from_id(self,id):
        row = self.intrinsics[id]
        fx = float(row[0])
        fy = float(row[4])
        cx = float(row[6])
        cy = float(row[7])

        return CameraModel.build_K_matrix_with_params(fx,fy,cx,cy)

