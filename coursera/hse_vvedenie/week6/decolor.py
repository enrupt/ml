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


def mean_color(cluster):
    cl_r, cl_g, cl_b = to_rgb(cluster)
    res = [None, None, None]
    res[0] = np.mean(cl_r)
    res[1] = np.mean(cl_g)
    res[2] = np.mean(cl_b)
    return res


def avg_color(cluster):
    cl_r, cl_g, cl_b = to_rgb(cluster)
    res = [None, None, None]
    res[0] = np.average(cl_r)
    res[1] = np.average(cl_g)
    res[2] = np.average(cl_b)
    return res


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


image_float = img_as_float(image)

r_arr = image_float[:, :, 0].ravel()
g_arr = image_float[:, :, 1].ravel()
b_arr = image_float[:, :, 2].ravel()
X = []
y = []
for i in range(0, len(r_arr)):
    y.append(i)
    X.append([r_arr[i], g_arr[i], b_arr[i]])

for n_clusters in range(8,21):
    kmeans = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters)
    predict = kmeans.fit_predict(X)

    clusters = [[]]
    for i in range(0, n_clusters):
        clusters.append([])

    for i in range(0, len(predict)):
        clusters[predict[i]].append(X[i])

    mean_cluster_colors = [[]]
    avg_cluster_colors = [[]]
    for i in range(0, n_clusters):
        mean_cluster_colors.append([])
        avg_cluster_colors.append([])

    for i in range(0, n_clusters):
        mean_cluster_colors[i] = mean_color(clusters[i])
        avg_cluster_colors[i] = avg_color(clusters[i])

    X_mean = np.empty((len(X), 3))
    X_avg = np.empty((len(X), 3))

    for i in range(0, len(X)):
        X_mean[i] = np.asarray(mean_cluster_colors[predict[i]])
        X_avg[i] = np.asarray(avg_cluster_colors[predict[i]])

    X_mean *= 255
    X_mean = X_mean.astype(np.uint8)
    X_mean_3d = np.reshape(X_mean, (height, width, -1))

    X_avg *= 255
    X_avg = X_avg.astype(np.uint8)
    X_avg_3d = np.reshape(X_avg, (height, width, -1))

    psnr_mean = psnr(X_mean_3d, image)
    psnr_avg = psnr(X_avg_3d, image)
    diff = psnr(X_mean_3d, X_avg_3d)
    print("clusters", n_clusters, psnr_mean, psnr_avg, diff)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    ax = axes.ravel()
    ax[0].set_title('Original picture')
    ax[0].imshow(image)
    ax[0].set_axis_off()

    ax[1].set_title('Mean picture')
    ax[1].imshow(X_mean_3d)
    ax[1].set_axis_off()

    ax[2].set_title('Avg picture n_clusters='+str(n_clusters))
    ax[2].imshow(X_avg_3d)
    ax[2].set_axis_off()

    plt.show()
