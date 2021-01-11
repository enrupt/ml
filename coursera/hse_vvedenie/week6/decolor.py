from skimage import io, filters, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

image = io.imread('parrots.jpg')

height = image.shape[0]
width = image.shape[1]
#print("shape", height, width)

# pylab.imshow(image)
# plt.show()


def to_rgb(cluster):
    arr_r = np.empty(len(cluster))
    arr_g = np.empty(len(cluster))
    arr_b = np.empty(len(cluster))
    for i in range(0, len(cluster)):
        arr_r[i] = cluster[i][0]
        arr_g[i] = cluster[i][1]
        arr_b[i] = cluster[i][2]
    return arr_r, arr_g, arr_b


def median_color(cluster):
    cl_r, cl_g, cl_b = to_rgb(cluster)
    mean_res = [None, None, None]
    mean_res[0] = np.median(cl_r)
    mean_res[1] = np.median(cl_g)
    mean_res[2] = np.median(cl_b)
    return mean_res


def avg_color(cluster):
    cl_r, cl_g, cl_b = to_rgb(cluster)
    avg_res = [None, None, None]
    avg_res[0] = np.average(cl_r)
    avg_res[1] = np.average(cl_g)
    avg_res[2] = np.average(cl_b)
    return avg_res


def mse(original, compressed):
    return np.mean((original - compressed) ** 2, dtype=np.float64)


def psnr(image_true, image_test):
    err = mse(image_true, image_test)
    return 10 * np.log10((np.max(image_true) ** 2) / err)


image_float = img_as_float(image)

X = np.array(image_float.reshape((image_float.shape[0]*image_float.shape[1], 3)))

for n_clusters in range(10, 20):
    kmeans = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters)
    predict = kmeans.fit_predict(X)

    clusters = [[]]
    for i in range(0, n_clusters):
        clusters.append([])

    for i in range(0, len(predict)):
        clusters[predict[i]].append(X[i])

    median_cluster_colors = [[]]
    avg_cluster_colors = [[]]
    for i in range(0, n_clusters):
        median_cluster_colors.append([])
        avg_cluster_colors.append([])

    for i in range(0, n_clusters):
        median_cluster_colors[i] = median_color(clusters[i])
        avg_cluster_colors[i] = avg_color(clusters[i])

    X_median = np.empty((len(X), 3))
    X_avg = np.empty((len(X), 3))

    for i in range(0, len(X)):
        X_median[i] = np.asarray(median_cluster_colors[predict[i]])
        X_avg[i] = np.asarray(avg_cluster_colors[predict[i]])

    X_median_3d = X_median*255
    X_median_3d = X_median_3d.astype(np.uint8)
    X_median_3d = np.reshape(X_median_3d, (height, width, -1))

    X_avg_3d = X_avg*255
    X_avg_3d = X_avg_3d.astype(np.uint8)
    X_avg_3d = np.reshape(X_avg_3d, (height, width, -1))

    psnr_median = psnr(X, X_median)
    psnr_avg = psnr(X, X_avg)
    diff = psnr(X_median, X_avg)
    print("clusters", n_clusters, "psnr_median", psnr_median, "psnr_avg", psnr_avg, mse(X, X_avg))

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    ax = axes.ravel()
    ax[0].set_title('Original picture')
    ax[0].imshow(image)
    ax[0].set_axis_off()

    ax[1].set_title('Mean picture')
    ax[1].imshow(X_median_3d)
    ax[1].set_axis_off()

    ax[2].set_title('Avg picture n_clusters='+str(n_clusters))
    ax[2].imshow(X_avg_3d)
    ax[2].set_axis_off()

   # plt.show()
