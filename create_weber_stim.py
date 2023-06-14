from cgi import test
import math
import os
import errno
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import itertools
import random

from torch import rand

from PIL import Image, ImageDraw
import math


class DashedImageDraw(ImageDraw.ImageDraw):

    def thick_line(self, xy, direction, fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        #direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill = fill, width = width)
        else:
            x1, y1 = xy[0]            
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = - (dx2 - dx1)/(dy2 - dy1)
                    a = 1/math.sqrt(1 + k**2)
                    b = (width*a - 1) /2
                else:
                    k = 0
                    b = (width - 1)/2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k*b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k*b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1)/2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1)/2)
            self.line([(x3, y3), (x4, y4)], fill = fill, width = 1)
        return   
        
    def dashed_line(self, xy, dash=(2,2), fill=None, width=0):
        #xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length**2 + y_length**2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion/length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line([(round(x1 + start*x_length),
                                          round(y1 + start*y_length)),
                                         (round(x1 + end*x_length),
                                          round(y1 + end*y_length))],
                                        xy, fill, width)
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2,2), outline=None, width=0):
        #xy - Sequence of [(x1, y1), (x2, y2)] where (x1, y1) is top left corner and (x2, y2) is bottom right corner
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        halfwidth1 = math.floor((width - 1)/2)
        halfwidth2 = math.ceil((width - 1)/2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1)%2        
        self.dashed_line([(x1 - halfwidth1, y1), (x2 - end_change1, y1)],
                         dash, outline, width)       
        self.dashed_line([(x2, y1 - halfwidth1), (x2, y2 - end_change1)],
                         dash, outline, width)
        self.dashed_line([(x2 + halfwidth2, y2 + odd_width_change),
                          (x1 + end_change2, y2 + odd_width_change)],
                         dash, outline, width)
        self.dashed_line([(x1 + odd_width_change, y2 + halfwidth2),
                          (x1 + odd_width_change, y1 + end_change2)],
                         dash, outline, width)
        return

def draw_line(length, width_range, lum_range, len_var, unit, pdash, size_imx, size_imy):
    im = Image.new('RGB', (size_imx, size_imy), color='black')

    ### Randomly draw width and luminance from range
    width = np.random.randint(width_range[0], width_range[1])
    lum = np.random.randint(lum_range[0], lum_range[1])
    delta_len = np.random.randint(len_var[0], len_var[1])
    len_in_pix = length * unit
    new_len_in_pix = len_in_pix + delta_len

    ### Find coordinates of line in the middle of image
    xc = size_imx/2
    yc = size_imy/2
    x0, y0 = xc - (new_len_in_pix / 2), yc
    x1, y1 = xc + (new_len_in_pix / 2), yc
    bbox = [(x0, y0), (x1, y1)]
    
    ### Draw the line
    if np.random.sample() > pdash: # sometimes draw solid line
        drawing = ImageDraw.Draw(im)
        drawing.line(bbox, width=width, fill=(lum,lum,lum))
    else: # other times draw dashed line
        dashed_drawing = DashedImageDraw(im)
        dashed_drawing.dashed_line(bbox, width=width, fill=(lum,lum,lum), dash=(2,2))

    return im


def draw_brightness(brightness, bright_unit, offset, radius_range, size_imx, size_imy):
    ### Create image with black as background
    im = Image.new('L', (size_imx, size_imy), color=0)
    drawing = ImageDraw.Draw(im)

    ### Convert brightness (percent) into a gray value [0-256)
    im_color = (brightness * bright_unit) + offset # scales 'brightness' but still in percent
    gray_val = int((im_color / 100) * 255)

    radius = np.random.randint(radius_range[0], radius_range[1]) # randomly sample a radius from the range

    ### Determine coordinates of stimulus "disk"
    xc = size_imx/2
    yc = size_imy/2
    x0, y0 = xc - radius, yc - radius
    x1, y1 = xc + radius, yc + radius
    bbox = [(x0, y0), (x1, y1)]

    ### Draw an stimulus disk of brightness gray_val
    drawing.ellipse(bbox, outline=None, fill=gray_val)

    return im


def gen_stim(stim_type='length', category=1, trans=(0,0), params=None):
    '''
    Generate a stimulus to test Weber's law
        stim_type = 'length' / 'brightness' / 'numerosity'
        category = stimulus category (determines intensity)
        params = various parameters of image / shape generation
    '''
    size_imx = params['size_x']
    size_imy = params['size_y']

    if stim_type == 'length':
        im = draw_line(length=category, width_range=params['line_width_range'], lum_range=params['line_lum_range'], len_var=params['length_var'], unit=params['length_unit'], pdash=params['pdash'], size_imx=size_imx, size_imy=size_imy)
        imt = translate_rotate(im, trans=trans, max_rot=params['max_rot'], flip90=params['flip90']) # transformed image
    elif stim_type == 'brightness':
        im = draw_brightness(brightness=category, bright_unit=params['bright_unit'], offset=params['offset'], radius_range=params['radius_range'], size_imx=size_imx, size_imy=size_imy)
        imt = translate_rotate(im, trans=trans, max_rot=params['max_rot'], fillcolor='black') # transformed image
    if stim_type == 'numerosity':
        im = Image.new('RGB', (size_imx, size_imy), color='white')
        drawing = ImageDraw.Draw(im)
        draw_line(drawing=drawing, length=category, width_range=params['line_width_range'], unit=params['length_unit'], xc=size_imx/2, yc=size_imy/2)
        imt = translate_rotate(im, trans=trans, max_rot=params['max_rot']) # transformed image
       
    return imt

def get_train_test_split(max_trans, ntrain, ntest):
    ### Generate a list of all possible locations based on translations
    xvals = list(range(-max_trans[0], max_trans[0])) # All x translations
    yvals = list(range(-max_trans[1], max_trans[1])) # All y translations
    all_locs = list(itertools.product(xvals, yvals)) # Generates all possible combinations of xvals and yvals

    train_test = random.sample(all_locs, ntrain+ntest) # Sample (without replacement) ntrain + ntest unique locations
    train = train_test[:ntrain]
    test = train_test[ntrain:]

    return (train, test)


def translate_rotate(im, trans, max_rot, fillcolor='black', flip90=False):
    ''' Generate a random translation and rotation of image
    '''
    if flip90 == True:
        ### Randomly chooose between 0 and 90 degree flip
        flip = np.random.randint(0,2) # flip is either 0 or 1
        theta = flip * 90
    else:
        theta = np.random.randint(-max_rot, max_rot+1) # random rotation
    tx_ii = trans[0] # random translation
    ty_ii = trans[1]

    ### First translate, then rotate - to ensure rotated segment remains completely on the canvas
    im = im.rotate(angle=0, translate=(tx_ii, ty_ii), resample=Image.BILINEAR, fillcolor=fillcolor)
    newim = im.rotate(angle=theta, resample=Image.BILINEAR, fillcolor=fillcolor)
    return newim


def get_params():
    params = {}

    ### Image related parameters
    scale = 1 # 1 means 224 x 224
    params['scale'] = scale
    params['size_x'] = scale * 224
    params['size_y'] = scale * 224

    ### Stimuli related parameters
    params['stim_types'] = ['length'] # ['length', 'brightness', 'numerosity']
    params['decoder_type'] = 'regr' # 'regr' / 'class'

    ### Intensity related parameters for testing Weber's law
    min_range = -4 # ncat = max_range - min_range
    max_range = 6
    params['test_intens'] = [5, 10, 15] # Mean intensities at which variation is compared
    # params['categories'] = [[ii + jj for jj in range(min_range, max_range)] for ii in params['test_intens']] # Something like [[1,..10], [6,..15], [11,..20]]
    params['categories'] = [(2*ii)+1 for ii in range(10)]

    # Parameters related to testing 'length'
    params['length_unit'] = 8 # pixels

    ### Parmeters related to testing 'brightness'
    params['bright_unit'] = 3 # luminosity expressed as percent (20% becomes 80%)
    params['offset'] = 20 # minimum brightness (some studies (e.g. Barlow, 1956) show Weber's Law doesn't hold for very dark values)
    params['radius_range'] = [int(params['size_x'] / 20), int(params['size_x'] / 6)] # [min, max) radius of patch

    ###  Following parameters are introduced to increase data variability along irrelevant (for length) dimensions
    params['line_width_range'] = [1, 5] # range of line widths in pixels [low, high)
    params['line_lum_range'] = [100, 256] # range of brightness values in range 0--255 [low, high)
    params['length_var'] = [0, 1] # range of length variability [low, high) in pixels (Note: this should be less than 'length_unit')
    params['max_rot'] = 0 # image will be rotated by random rotation in the range [-max_rot, +max_rot] (degrees)
    params['flip90'] = False # instead of any rotation (max_rot), train 0 & 90 degrees
    params['pdash'] = 0 # probability of dashed line (rather than solid)

    if params['stim_types'][0] == 'length':
        params['max_trans'] = [int((params['size_x'] / 2) - ((params['categories'][-1] / 2) * params['length_unit'] + params['line_width_range'][-1])), int((params['size_y'] / 2) - params['line_width_range'][-1])]
    elif params['stim_types'][0] == 'brightness':
        params['max_trans'] = [int(params['size_x'] / 2) - params['radius_range'][1], int(params['size_y'] / 2) - params['radius_range'][1]]

    ### Simulation related parameters
    params['ntrain'] = 5000 # 5000 number of training examples for each shape
    params['ntest'] = 200 # 200 number of training examples for each shape

    return params


def create_set(params, set_type='train'):
    stim_types = params['stim_types']
    ntrain = params['ntrain']
    ntest = params['ntest']
    intensities = params['test_intens']
    (train_samples, test_samples) = get_train_test_split(params['max_trans'], ntrain=ntrain, ntest=ntest)

    for (stim_id, stim_name) in enumerate(stim_types):
        print("Creating {0} images for {1} stimuli             ".format(set_type, stim_name), end="\r")
        if set_type == 'train':
            niter = ntrain
        elif set_type == 'test': # only change niter, other params same as training
            niter = ntest

        # for (intens_id, intens_val) in enumerate(intensities):
        cats = params['categories']
        for cc in cats:
            for ii in range(niter):
                if set_type == 'train':
                    trans = train_samples[ii]
                elif set_type == 'test':
                    trans = test_samples[ii]
                im = gen_stim(stim_type=stim_name, category=cc, trans=trans, params=params)

                # filename = os.path.join('mind-set', 'data' , stim_name, str(intens_val), set_type, str(cc).zfill(2), str(ii) + '.png')
                if params['decoder_type'] == 'class':
                    filename = os.path.join('mind-set', 'data' , stim_name, set_type, str(cc).zfill(2), str(ii) + '.png')
                elif params['decoder_type'] == 'regr':
                    filename = os.path.join('mind-set', 'data' , stim_name, set_type, str(cc) + '.' + str(ii) + '.png')
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                im.save(filename)

create_set(get_params(), 'train')
create_set(get_params(), 'test')
print("\n")
