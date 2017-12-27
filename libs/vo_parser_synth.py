import csv
import numpy.matlib as matlib
import libs.sdk.python.transform as transform
import math
import numpy as np
import libs.transformations

"""Parses a supplied text file with SE3 parameters.
   SE3 Matricies are absolute and not relative between frames

  Attributes:
      vo_path: the path of the file
      data_string: raw data read from the file
      vo_headers: csv headers 
      vo_data: parsed vo data. i.e. relevant data is converted into approiate data type. e.g. int, float
      vo_index_list: mapping form source timestamp to index of vo_data
      
      LEFT HANDED COORDINATE SYSTEM (OPEN CV STYLE)
  """


class VoParserSynth:
    def __init__(self, path):

        self.vo_path = path

        self.vo_data_string = []  # raw data
        self.vo_headers = []
        self.vo_data = []  # data converted to appropriate structures i.e. int, float
        self.vo_index_dictionary = {}  # mapping form source timestamp to index of vo_data
        self.load_csv()

    def load_csv(self):

        file = open(self.vo_path, 'r')
        reader = csv.reader(file, delimiter=' ')

        for row in reader:
            self.vo_data_string.append(row)

        vo_data_length = len(self.vo_data_string)

        for i in range(0, vo_data_length,1):
            data_raw = self.vo_data_string[i]
            id = int(data_raw[0])
            data = (
                id,  #image id
                float(data_raw[1]),  # x
                float(data_raw[2]),  # y
                float(data_raw[3]),  # z
                float(data_raw[4]),  # qx
                float(data_raw[5]),  # qy
                float(data_raw[6]),  # qz
                float(data_raw[7])  # qw
            )
            self.vo_data.append(data)
            self.vo_index_dictionary[id] = i

        file.close()

        return (self.vo_headers, self.vo_data)

    '''Returns 4x4 SE3 Matrix'''
    def buildSE3(self, sourceImageId, destImageId):

        if (len(self.vo_data) == 0):
            raise AttributeError("No CSV data was read. Call loadcsv() first")

        indexToLookUp = self.vo_index_dictionary[sourceImageId]
        (id, x_source, y_source, z_source, qx_source, qy_source, qz_source,qw_source) = self.vo_data[indexToLookUp]

        (roll_source,pitch_source,yaw_source) = VoParserSynth.Quaternion_toEulerianRadians(qx_source, qy_source, qz_source, qw_source) # use library

        indexToLookUp = self.vo_index_dictionary[destImageId]
        (id, x_dest, y_dest, z_dest, qx_dest, qy_dest, qz_dest,qw_dest) = self.vo_data[indexToLookUp]

        (roll_dest, pitch_dest,yaw_dest) = VoParserSynth.Quaternion_toEulerianRadians(qx_dest, qy_dest, qz_dest, qw_dest)

        R_1_c_w = VoParserSynth.makeS03(roll_source,pitch_source,yaw_source)
        R_1_w_c = np.transpose(R_1_c_w)
        R_2_c_w = VoParserSynth.makeS03(roll_dest,pitch_dest,yaw_dest)
        R_2_w_c = np.transpose(R_2_c_w)

        T_1_w = np.array([[0],[0],[0]],float)
        T_1_w[0,0] = x_source
        T_1_w[1,0] = y_source
        T_1_w[2,0] = z_source

        T_2_w = np.array([[0],[0],[0]],float)
        T_2_w[0,0] = x_dest
        T_2_w[1,0] = y_dest
        T_2_w[2,0] = z_dest

        diff_w = T_2_w - T_1_w
        T_1_2_prime = np.matmul(R_1_w_c,diff_w)

        x_prime = T_1_2_prime[0,0]
        y_prime = T_1_2_prime[1,0]
        z_prime = T_1_2_prime[2,0] # positive
        # x_prime *= -1
        # y_prime *= -1
        # z_prime *= -1

        R_c1_c2 = np.matmul(R_2_w_c,R_1_c_w)

        se3 = transform.build_se3_transform([x_prime,y_prime,z_prime,0,0,0]) #TODO transform call not needed

        se3[0:3,0:3] = R_c1_c2[0:3,0:3]

        se3_source_to_dest = np.identity(4)

        se3_source_to_dest[:,:] = se3[:,:]

        return se3_source_to_dest


    #https: // en.wikipedia.org / wiki / Conversion_between_quaternions_and_Euler_angles
    @staticmethod
    def Quaternion_toEulerianRadians(x_raw, y_raw, z_raw, w_raw):

        n =  math.sqrt(x_raw*x_raw+y_raw*y_raw+z_raw*z_raw+w_raw*w_raw)
        x = x_raw
        y = y_raw
        z = z_raw
        w = w_raw

        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z



    @staticmethod
    def makeS03(roll_rad,pitch_rad,yaw_rad):
        R_x = np.identity(3)
        R_y = np.identity(3)
        R_z = np.identity(3)

        R_x[1,1] = math.cos(roll_rad)
        R_x[1,2] = -math.sin(roll_rad)
        R_x[2,1] = math.sin(roll_rad)
        R_x[2,2] = math.cos(roll_rad)

        R_y[0,0] = math.cos(pitch_rad)
        R_y[0,2] = math.sin(pitch_rad)
        R_y[2,0] = -math.sin(pitch_rad)
        R_y[2,2] = math.cos(pitch_rad)

        R_z[0,0] = math.cos(yaw_rad)
        R_z[0,1] = -math.sin(yaw_rad)
        R_z[1,0] = math.sin(yaw_rad)
        R_z[1,1] = math.cos(yaw_rad)

        S03 = np.matmul(R_z,np.matmul(R_y,R_x))

        return S03


    @staticmethod
    def makeSE3(x,y,z,qx,qy,qz,qw):
        n =  math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
        qx_norm = qx/n
        qy_norm = qy/n
        qz_norm = qz/n
        qw_norm = qw/n

        se3 = matlib.identity(4)

        se3[0,0] = 1-2*qy_norm*qy_norm-2*qz_norm*qz_norm
        se3[1,0] =2*qx_norm*qy_norm + 2*qz_norm*qw_norm
        se3[2, 0] = 2*qx_norm*qz_norm - 2*qy_norm*qw_norm
        se3[0,1] = 2*qx_norm*qy_norm - 2*qz_norm*qw_norm
        se3[1,1] = 	1 - 2*qx_norm*qx_norm - 2*qz_norm*qz_norm
        se3[2, 1] = 2*qy_norm*qz_norm + 2*qx_norm*qw_norm
        se3[0,2] = 2*qx_norm*qz_norm + 2*qy_norm*qw_norm
        se3[1,2] = 2*qy_norm*qz_norm - 2*qx_norm*qw_norm
        se3[2, 2] = 1 - 2*qx_norm*qx_norm - 2*qy_norm*qy_norm

        se3[0,3] = x
        se3[1,3] = y
        se3[2,3] = z

        return se3



