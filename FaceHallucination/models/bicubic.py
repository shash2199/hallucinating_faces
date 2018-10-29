import glob
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave

def bicubic(im):
    # im: 2-D numpy array (32x24)
    return imresize(im, (128, 96))

target_files = glob.glob('../profile_pictures_aligned/*.png')
target_ims = [imresize(imread(fn, mode='L'), (32, 24)) for fn in target_files]

for i, target_im in enumerate(target_ims):
    result = bicubic(target_im)
    imsave('../bicubic_results/{0}.png'.format(target_files[i]), result)
