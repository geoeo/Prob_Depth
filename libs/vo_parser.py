import csv
import numpy.matlib as matlib
import libs.sdk.python.transform as transform

"""Parses a supplied text file with SE3 parameters between consecutive image frames.

  Attributes:
      vo_path: the path of the file
      data_string: raw data read from the file
      vo_headers: csv headers 
      vo_data: parsed vo data. i.e. relevant data is converted into approiate data type. e.g. int, float
      vo_index_list: mapping form source timestamp to index of vo_data
  """


class VoParser:
    def __init__(self, path):

        self.vo_path = path

        self.vo_data_string = []  # raw data
        self.vo_headers = []
        self.vo_data = []  # data converted to appropriate structures i.e. int, float
        self.vo_index_dictionary = {}  # mapping form source timestamp to index of vo_data
        self.load_csv()

    def load_csv(self):

        file = open(self.vo_path, 'r')
        reader = csv.reader(file)

        for row in reader:
            self.vo_data_string.append(row)

        vo_data_length = len(self.vo_data_string)

        self.vo_headers.append(self.vo_data_string[0])

        for i in range(1, vo_data_length,1):
            data_raw = self.vo_data_string[i]
            ts_0 = int(data_raw[0])
            data = (
                ts_0,  # timestamp 0 - source
                int(data_raw[1]),  # timestamp 1 - destination
                float(data_raw[2]),  # x
                float(data_raw[3]),  # y
                float(data_raw[4]),  # z
                float(data_raw[5]),  # roll
                float(data_raw[6]),  # pitch
                float(data_raw[7])  # raw
            )
            self.vo_data.append(data)
            self.vo_index_dictionary[ts_0] = i - 1

        file.close()

        return (self.vo_headers, self.vo_data)

    def buildSE3(self, sourceTimeStamp, destTimeStamp, SE3=matlib.identity(4)):

        if (len(self.vo_data) == 0):
            raise AttributeError("No CSV data was read. Call loadcsv() first")

        indexToLookUp = self.vo_index_dictionary[sourceTimeStamp]
        (source, dest, x, y, z, r, p, y) = self.vo_data[indexToLookUp]
        se3 = transform.build_se3_transform([x, y, z, r, p, y])
        se3 = se3 * SE3

        if (destTimeStamp == dest):
            return se3
        else:
            return self.buildSE3(dest, destTimeStamp, SE3=se3)
