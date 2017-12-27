def write_pcd_header(f, num_points):
    f.write(
        "# .PCD v.7 - Point Cloud Data file format\n"
        "VERSION .7\n"
        "FIELDS x y z rgba\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        "WIDTH " + str(num_points) + "\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS " + str(num_points) + "\n"
        "DATA ascii\n")

def _write_pcd(fn, xyzRGB):
    with open(fn,"w") as f:
       write_pcd_header(f, xyzRGB.shape[1])

       for row in xyzRGB.T:
           f.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' +
                   str(row[3] + row[4]*2**8 + row[5]*2**16) + "\n")                

def write_pcd(path, frameIdx, xyzRGB):
    print("write " + frameIdx + " ...")
    fn = path + str(frameIdx) + ".pcd"
    _write_pcd(fn,xyzRGB)
    
def write_ply_header(f, num_points):
    f.write(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex " + str(num_points) + "\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n")


def _write_ply(fn, xyzRGB):
    with open(fn, "w") as f:
        write_ply_header(f, xyzRGB.shape[1])

        for row in xyzRGB.T:
            f.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' +
                    str(int(row[3])) + ' ' + str(int(row[4])) + ' ' + str(int(row[5])) + "\n")
                    #str(row[6]) + ' ' + str(row[7]) + ' ' + str(row[8]) + "\n")


def write_ply(path, frameIdx, xyzRGB):
    print("write " + frameIdx + " ...")
    fn = path + str(frameIdx) + ".ply"
    _write_ply(fn, xyzRGB)

