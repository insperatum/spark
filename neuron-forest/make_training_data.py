import numpy as np
from scipy import ndimage, misc, io
from matplotlib import pyplot
import os
import h5py

def makeFeatures(img, filename):
    orders = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 1], [0, 2, 0], [1, 1, 0], [2, 0, 0]]
    scales = [1, 2, 4]

    print("Creating features: " + filename)
    #NOTE: force big-endian for use at scala end!
    features = np.empty((np.prod(img.shape), len(orders) * len(scales)), dtype=">f")
    i = 0
    for scale in scales:
        print("  Scale " + str(scale))
        for o in orders:
            print("    Order " + str(o))
            features[:, i] = ndimage.filters.gaussian_filter(img, scale, o).flatten(order = 'C')
            i += 1

    print("  Saving")
    features.tofile(filename + ".raw")
    #np.savetxt(filename + ".txt", features, fmt='%.6f')
    #io.savemat(filename + ".mat", {'features':features})

def makeTargets(segTrue, bounds, filename):
    print("Creating targets: " + filename)
    min_idx = bounds[:, 0]
    max_idx = bounds[:, 1]-1 # -1 because no affinity on faces
    idxs = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx)
    targets = get_target_affinities(segTrue, idxs).astype(np.int32)
    print("  Saving")
    np.savetxt(filename + ".txt", targets, fmt='%d')

def makeDimensions(im, bounds, filename):
    print("Creating dimensions: " + filename)
    file = open(filename + ".txt", 'w')
    min_idx = bounds[:, 0]
    max_idx = bounds[:, 1]-1 # -1 because no affinity on faces
    file.write(" ".join([str(i) for i in im.shape]) + "\n")
    file.write(" ".join([str(i) for i in min_idx]) + "\n")
    file.write(" ".join([str(i) for i in max_idx]))
    file.close()

# -------------------------------------------------
def get_steps(arr):
    return tuple(np.append(np.cumprod(np.array(arr.shape)[1:][::-1])[::-1], 1))

def get_image_idxs(im, max_idx, min_idx=(0,0,0)):
    xs, ys, zs = np.ix_(range(min_idx[0], max_idx[0] + 1), range(min_idx[1], max_idx[1] + 1),
                    range(min_idx[2], max_idx[2] + 1))
    steps = get_steps(im)
    return np.array(np.unravel_index((xs * steps[0] + ys * steps[1] + zs * steps[2]).flatten(), im.shape))

def get_target_affinities(seg, idxs):
    aff = np.empty((len(idxs[0]), 3), dtype=bool)
    aff[:, 0] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[1], [0], [0]])])
    aff[:, 1] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [1], [0]])])
    aff[:, 2] = np.logical_and(seg[tuple(idxs)] != 0, seg[tuple(idxs)] == seg[tuple(idxs + [[0], [0], [1]])])
    return aff


# --------------------------------------------------

print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat("/home/luke/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")
if not os.path.exists("data"): os.mkdir("data")
for i in range(0, 1):
    folder = "data/im" + str(i+1)
    if not os.path.exists(folder): os.mkdir(folder)
    makeFeatures(Helmstaedter2013["im"][0, i], folder + "/features")
    makeTargets(Helmstaedter2013["segTrue"][0, i], Helmstaedter2013["boundingBox"][0, i], folder + "/targets")
    makeDimensions(Helmstaedter2013["im"][0, i], Helmstaedter2013["boundingBox"][0, i], folder + "/dimensions")