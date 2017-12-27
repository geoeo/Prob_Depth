import numpy as np
import csv

class VoParserSynth:
    def __init__(self, path,sequence):
        self.vo_path = path # base_# dir
        self.sequence = sequence
        self.vo_data_string = [] # one string entry represents 4x4 column major matrix
        self.load_pose_file()

    def load_pose_file(self):
        file = open(self.vo_path+self.sequence+"poses.txt", 'r')
        reader = csv.reader(file, delimiter=' ')

        for row in reader:
            #if not row.isspace() and row TODO investigate this for parsing input
            self.vo_data_string.append(row)


    def buildSE3(self, sourceImageId, destImageId):

        SE3_source_elements = self.vo_data_string[sourceImageId]
        SE3_dest_elemens = self.vo_data_string[destImageId]

        SE3_source = self.createMatrix(SE3_source_elements)
        SE3_dest = self.createMatrix(SE3_dest_elemens)

        R_source_trans = np.transpose(SE3_source[0:3, 0:3])
        R_source_dest = np.matmul(SE3_dest[0:3, 0:3], R_source_trans)

        t_source_c = np.matmul(R_source_trans,SE3_source[0:3,3])
        t_dest_c = np.matmul(R_source_trans,SE3_dest[0:3,3])
        t_source_dest = t_dest_c - t_source_c
        # t_source_dest = SE3_dest[0:3,3] - SE3_source[0:3,3]
        # t_source_dest[0]*= -1
        t_source_dest[1]*= -1
        # t_source_dest[2]*= -1

        se3 = np.identity(4)
        se3[0:3,0:3] = R_source_dest
        se3[0:3,3] = t_source_dest


        return se3

    def createMatrix(self,elements):

        matrix = np.identity(4)

        matrix[0,0] = float(elements[0])
        matrix[1,0] = float(elements[1])
        matrix[2,0] = float(elements[2])
        matrix[3,0] = float(elements[3])

        matrix[0,1] = float(elements[4])
        matrix[1,1] = float(elements[5])
        matrix[2,1] = float(elements[6])
        matrix[3,1] = float(elements[7])

        matrix[0,2] = float(elements[8])
        matrix[1,2] = float(elements[9])
        matrix[2,2] = float(elements[10])
        matrix[3,2] = float(elements[11])

        matrix[0,3] = float(elements[12])
        matrix[1,3] = float(elements[13])
        matrix[2,3] = float(elements[14])
        matrix[3,3] = float(elements[15])

        return matrix




