import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.ndimage.filters import convolve
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from tqdm import tqdm

# Sobel filters
first_sobel_horiz = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

first_sobel_vert = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# https://dsp.stackexchange.com/a/10622
second_sobel_horiz = np.array([
    [1, -2, 1],
    [2, -4, 2],
    [1, -2, 1]
])

second_sobel_vert = np.array([
    [1, 2, 1],
    [-2, -4, -2],
    [1, 2, 1]
])

def W(m, n, p, q, k):
    if m * 2**k <= p < (m+1) * 2**k and n * 2**k <= q < (n+1) * 2**k:
        return 1 / 2 ** (2*k)
    else:
        return 0

def map_formulation(G_0, G_k, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_k | G_0]
    # G_0 - predicted first image of Gaussian pyramid
    # G_k - image at the kth level of the Gaussian pyramid
    return 1 / (2*covariance) * sum(
        sum(G_k[m, n] - sum(
            sum(
                W(m, n, p, q, k) * G_0[p, q]
                for q in range(width*2**k))
            for p in range(height*2**k))
            for n in range(width))
        for m in range(height))

def gradient_prior(H0_G0, V0_G0, H0_I, V0_I, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_0]
    # H0_G0 - horizontal derivative of G_0
    # V0_G0 - vertical derivative of G_0
    # H0_I - predicted horizontal derivative of G_0
    # V0_I - predicted vertical derivative of G_0
    return 1 / (2*covariance) * sum(
        sum(
            sum((H0_G0[m, n] - H0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height))) + 1 / (2*covariance) * sum(
        sum(
            sum((V0_G0[m, n] - V0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height)))

def F(g_layer, l_layer):
    # create feature vector:
    # (laplacian, horizontal derivative, vertical derivative, 2nd h.d., 2nd v.d.)
    return (
        l_layer, convolve(g_layer, first_sobel_horiz), convolve(g_layer, first_sobel_vert),
        convolve(g_layer, second_sobel_horiz), convolve(g_layer, second_sobel_vert)
    )

def F_idx(fv, m, n):
    return tuple(layer[m, n] for layer in fv)

def PS(im, depth):
    g_pyramid = pyramid_gaussian(im, max_layer=depth)
    l_pyramid = pyramid_gaussian(im, max_layer=depth)
    return [F(g_layer, l_layer) for g_layer, l_layer in zip(g_pyramid, l_pyramid)]

def PS_idx(ps, m, n):
    ret = []
    for fv in ps:
        ret.append(F_idx(fv, m, n))
        m //= 2
        n //= 2
    return ret

def predict_high_res(training_ims, low_res_im, k=2, N=4):
    low_res_fvs = PS(low_res_im, N-k)
    training_im_fvs = [PS(im, N)[:k] for im in training_ims]
    def PS_error(fvs, m, n):
        weights = np.array([1, 0.5, 0.5, 0.5, 0.5])
        error = 0
        low_res_pixels = PS_idx(low_res_fvs, m, n)
        fv_pixels = PS_idx(fvs, m, n)
        for i in range(N-k):
            error += np.linalg.norm(np.multiply(weights, np.subtract(low_res_pixels[i], fv_pixels[i])), ord=2)
            weights /= 2
        return error

    high_res = np.zeros((low_res_im.shape[0] * 2**k, low_res_im.shape[1] * 2**k))
    pbar = tqdm(total=low_res_im.size * 4**k)
    for m in range(low_res_im.shape[0] * 2**k):
        for n in range(low_res_im.shape[1] * 2**k):
            lr_m = min(round(m / 2**k), low_res_im.shape[0]-1)
            lr_n = min(round(n / 2**k), low_res_im.shape[1]-1)
            fv_errors = [PS_error(fvs, lr_m, lr_n) for fvs in training_im_fvs]
            j = np.argmin(fv_errors)
            high_res[m, n] = training_ims[j][m, n]
            pbar.update(1)
    pbar.close()
    return high_res

training_ims = glob.glob('../ucsd_aligned_images/*.png')
training_ims = [imread(fn, mode='L') for fn in training_ims]
target_files = glob.glob('../profile_pictures_aligned/*.png')
target_ims = [imresize(imread(fn, mode='L'), (32, 24)) for fn in target_files]

for i, target_im in enumerate(target_ims):
    predicted_im = predict_high_res(training_ims, target_im)
    imsave('../bayes_results/{0}.png'.format(target_files[i]), predicted_im)
    #plt.figure()
    #plt.imshow(predicted_im, cmap='gray')
#plt.show()
