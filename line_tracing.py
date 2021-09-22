#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sep 21 17:09:45 2021

@author: Nacriema

Refs:

Code for line tracing algorithm
Reused the code from Image_Gradients

Line tracing has 3 main ideas:
* Calculate all gradients needed
* Choose the initial point
* Use stack, queue to track the ray of line base on gradients
* Required Direction Constrain while doing searching (Currently, I do not implemented it)
* Refined the detection line by using Curve Fitting approximation

NOTE: This is not a full version, because this is lack of Searching Direction Constrain, I'll add it later. Because I
don't understand the concept - need more research.

In the code, I write the directly form to compute grad_45_deg and grad_neg_45_deg as well as another way to compute this
I strongly suggest using the latter one, because it can approximate the gradients at any directions given by theta.

I also add the direction constrain as the author mentioned in their paper. In my opinion, this addition can help with
the large image when some init point we not care to much about them, I said that because this can modify the correct
detected points near the init points, but overall, it can help detect the line better.


NOTICE: Some important parameter need to consider while using this algorithm:
* theta: is the angle between  ------> x axis
                                ` (theta > 0)
                                 `  power line
                                   `
         theta > 0: line bellow x, theta < 0: line above x
* Due to camera position and the environment around the camera, we can set the delta(y) / delta(x) < theta (theta > 0).
Or we can set the abs(delta(y)/delta(x)) < theta to make the predict line not go to low or to high to the x axis. Or you
can consider 0 < delta(y)/delta(x) < theta.

With image im_1.jpeg allow the line to go up, then choose the delta(y) / delta(x) < theta
                                                           or abs(delta(y)/delta(x)) < theta
With image im_7.jpg the line shape not go up, then choose 0 < delta(y)/delta(x) < theta
                                                       or abs(delta(y)/delta(x)) < theta

For better visualization, I'll choose different constrain for different condition. This is up to you in setting:
With img_1: theta 0.1, abs(delta(y)/delta(x)) < theta
With img_7: theta 0.05, 0 < delta(y)/delta(x) < theta

OVERALL:
Base on camera setting, these params need to take into account:
* theta
* gradient directions: we can add more directions to track the line in different orient
* Direction constrain: change inside the direction_constrain function
* Line tracing changing too as the gradient change.

"""

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from math import ceil
from PIL import Image
from scipy.optimize import curve_fit


def imshow_all(*images, titles=None, cmap=None, ncols=3):
    images = [img_as_float(img) for img in images]
    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=ceil(len(images) / ncols), ncols=ncols, figsize=(width, height), dpi=96)
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)


def partial_dev_along_x_2(image):
    horizontal_kernel = np.array([
        [-1, 0, 1]
    ])
    gradient_horizontal = ndi.correlate(image.astype(float),
                                        horizontal_kernel)
    return gradient_horizontal


def partial_dev_along_y_2(image):
    vertical_kernel = np.array([
        [-1],
        [0],
        [1],
    ])
    gradient_vertical = ndi.correlate(image.astype(float),
                                      vertical_kernel)
    return gradient_vertical

# TODO: Due to the lack of precision, because we need the exactly gradient form the raw image, then I implemented 2
#  kernel to detect grad_45 and grad_neg_45, replace it with create_directional_image_derivatives


def partial_dev_along_45_deg(image):
    kernel = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0]
    ])
    gradient_45_deg = ndi.correlate(image.astype(float), kernel)
    return gradient_45_deg


def partial_dev_along_neg_45_deg(image):
    kernel = np.array([
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    gradient_neg_45_deg = ndi.correlate(image.astype(float), kernel)
    return gradient_neg_45_deg


def create_mag_and_ori(mag, grad_x, grad_y):
    """
    Create RGB image that visualize magnitude and orientation simultaneously
    red(x,y) = |mag|.cos(theta), green(x, y) = |mag|.sin(theta), blue(x,y) = 0
    and
    theta = tan-1(grad_x/grad_y)
    :param grad_x:
    :param grad_y:
    :return: rgb in numpy format
    """
    theta = np.abs(np.arctan(grad_x / grad_y))
    green = mag * np.sin(theta)
    red = mag * np.cos(theta)
    blue = np.zeros_like(mag)
    rgb = np.dstack((red, green, blue))
    return rgb


def create_directional_image_derivatives(theta, grad_x, grad_y):
    """
    Compute the directional derivative with given the theta direction
    In matrix form, at the (0, 0) point for example:
                                   [cos(theta)
    [grad_x(0, 0) grad_y(0, 0)]. *
                                    sin(theta)]
    NOTICE: In this formula the coordinate as follow:
    ^y
    |   /
    |  /
    | /
    |/ theta
    . - - - - - ->x
     So when apply this function in image, give my result as the inversion of grad_45 and grad_neg_45, use it cautiously
     Just add the negative sign to the grad_y when calculate to fix it
    :param theta:
    :param grad_x:
    :param grad_y:
    :return: np array with shape like grad_x
    """
    return grad_x * np.cos(theta) - grad_y * np.sin(theta)


def save_mag_image(mag_and_ori):
    mag_and_ori = np.nan_to_num(mag_and_ori)
    print(np.unique(mag_and_ori))
    im = Image.fromarray(np.uint8(mag_and_ori * 255))
    im.save('mag_and_ori_2.jpg')


# LINE TRACING PART
'''
Notice about the notation in this code: 
. - - - - - - -> width 
|
|
|
|
|
V height


image pixel at w, h is image[h, w] (image is the np.array type)
'''


def direction_constrain(init_h, init_w, curr_h, curr_w, k0=0, threshold=0.05):
    """
    Check if the current interest point is ok or not by:
    - Calculate the slope k between current_point and init_point
    - Then determine whether the direction constrain is needed according to the condition: k - k0 < threshold
    :param init_h: height of init point
    :param init_w: width of init point
    :param curr_h: height of current point
    :param curr_w: width of current point
    :param k0:
    :param threshold:
    :return: True or False
    """
    # Special case:
    if init_h == curr_h:
        return True
    else:
        # Calculate slope k
        # TODO: NEED MORE CONSIDERATION, WELL THIS IS ALSO IMPACT WHEN CHOOSING SLOPE
        k = (curr_h - init_h) / (curr_w - init_w)
        # TODO: adjust here to h/w
        # print(f'curr_h = {curr_h}, curr_w={curr_w}, init_h={init_h}, ini_w={init_w}, k={k}')
        # return abs(k - k0) < threshold
        return 0 < k - k0 < threshold


def choose_interest_point_by_slope(init_h, init_w, curr_h, curr_w, init_set, k0=0):
    """
    Choose the interest point base on the slopes constrain
    :param init_h:
    :param init_w:
    :param curr_h:
    :param curr_w:
    :param init_set:
    :param k0:
    :return:
    """
    # Special case: Handle the special cases, may be we don't care too much about them
    if curr_h - init_h == 0:
        min_index = 0
    elif curr_h - 1 - init_h == 0:
        min_index = 1
    elif curr_h + 1 - init_h == 0:
        min_index = 2
    else:
        # TODO: Adjust here
        slope_0 = (curr_h - init_h)/(curr_w + 1 - init_w)
        slope_45 = (curr_h - 1 - init_h)/(curr_w + 1 - init_w)
        slope_neg_45 = (curr_h + 1 - init_h)/(curr_w + 1 - init_w)
        # TODO: Here is also can modify base on condition
        watch_list_slopes = [abs(slope_0 - k0), abs(slope_45-k0), abs(slope_neg_45-k0)]
        # watch_list_slopes = [slope_0 - k0, slope_45-k0, slope_neg_45-k0]
        min_index = watch_list_slopes.index(min(watch_list_slopes))
    if min_index == 0:
        init_set.append((curr_h, curr_w + 1))
    elif min_index == 1:
        init_set.append((curr_h - 1, curr_w + 1))
    else:
        init_set.append((curr_h + 1, curr_w + 1))
    return init_set


def line_tracing(init_point, grad_x, grad_45, grad_neg_45):
    init_set = [init_point]
    init_h, init_w = init_point
    result_set = []
    # Give the stop condition here
    while init_set:
        current_point = init_set.pop()
        # print(current_point)
        # Check reached image boundary here
        if check_valid(current_point, grad_x):
            result_set.append(current_point)
            curr_h, curr_w = current_point
            watch_list_grads = [grad_x[curr_h, curr_w], grad_45[curr_h, curr_w], grad_neg_45[curr_h, curr_w]]
            min_index = watch_list_grads.index(min(watch_list_grads))

            # TODO: Modified for testing, change to not using constrain
            if min_index == 0:
                if direction_constrain(init_h=init_h, init_w=init_w, curr_h=curr_h, curr_w=curr_w + 1):
                    init_set.append((curr_h, curr_w + 1))
                else:
                    print('Choose by slope')
                    init_set = choose_interest_point_by_slope(init_h=init_h, init_w=init_w, curr_h=curr_h, curr_w=curr_w
                                                              , init_set=init_set)
                    # init_set.append((curr_h, curr_w + 1))
            elif min_index == 1:
                if direction_constrain(init_h=init_h, init_w=init_w, curr_h=curr_h - 1, curr_w=curr_w + 1):
                    init_set.append((curr_h - 1, curr_w + 1))
                else:
                    print('Choose by slope')
                    init_set = choose_interest_point_by_slope(init_h=init_h, init_w=init_w, curr_h=curr_h, curr_w=curr_w
                                                              , init_set=init_set)
                    # init_set.append((curr_h - 1, curr_w + 1))
            else:
                if direction_constrain(init_h=init_h, init_w=init_w, curr_h=curr_h + 1, curr_w=curr_w + 1):
                    init_set.append((curr_h + 1, curr_w + 1))
                else:
                    print('Choose by slope')
                    init_set = choose_interest_point_by_slope(init_h=init_h, init_w=init_w, curr_h=curr_h, curr_w=curr_w
                                                              , init_set=init_set)
                    # init_set.append((curr_h + 1, curr_w + 1))
    return result_set


def generate_mask_im(source_im, result_set):
    """
    Given image and result set, reconstruct the mask image
    :param source_im:
    :param result_set:
    :return: mask image
    """
    # print('=============================s')
    result_images = np.zeros_like(source_im)
    for item in result_set:
        # print(item)
        result_images[item[0], item[1]] = 1
    return result_images


# TODO: Modified here, for checking
def check_valid(point_index, image_arr):
    return point_index[0] < image_arr.shape[0] and point_index[1] < image_arr.shape[1]


def objective(w, a, b, c):
    return a * w + b * w ** 2 + c


def use_curve_fitting(result_set):
    """
    Use built-in function of scipy for apply curve fitting step
    :param result_set:
    :return:
    """
    # print(result_set)
    w = np.array([i[1] for i in result_set])
    h = np.array([i[0] for i in result_set])
    popt, _ = curve_fit(objective, w, h)
    a, b, c = popt
    w_line = np.arange(min(w), max(w), 1)
    h_line = objective(w_line, a, b, c)
    h_line = np.around(h_line).astype(int)
    result = [(h_line[i], w_line[i]) for i in range(len(w_line))]
    # print("===================")
    # print(result)
    return result


def create_image_with_mask(image, result_set, color='red'):
    """
    Plot the detected line on the original image, default color is red
    :param image:
    :param result_set: contains x, y coordinate of
    :return: image with detected line
    """
    image_ = np.copy(image)
    for item in result_set:
        if color == 'red':
            image_[item[0], item[1]] = np.array([255, 0, 0])
        else:
            image_[item[0], item[1]] = np.array([0, 0, 255])
    return image_


if __name__ == '__main__':
    # im_1 (260, 258)
    # image_rgb = imread('./Images/Im_1.jpeg')
    # init_point = (260, 258)
    # im_7 (50, 191) and (206, 216) and (61, 192)
    image_rgb = imread('./Images/im_7.jpg')
    init_point = (61, 192)
    # print(image_rgb[74, 315])
    image = rgb2gray(image_rgb)
    # print(image.shape)
    x_grad = partial_dev_along_x_2(image)
    y_grad = partial_dev_along_y_2(image)
    g = np.sqrt(x_grad ** 2 + y_grad ** 2)
    mag_and_ori = create_mag_and_ori(g, x_grad, y_grad)
    '''
    We need 2 value more, the 45 degree and -45 degree directional derivative
    '''
    grad_45 = create_directional_image_derivatives(theta=(45 / 180) * (2 * np.pi), grad_x=x_grad, grad_y=y_grad)
    # grad_45 = partial_dev_along_45_deg(image)
    grad_neg_45 = create_directional_image_derivatives(theta=(-45 / 180) * (2 * np.pi), grad_x=x_grad, grad_y=y_grad)
    # grad_neg_45 = partial_dev_along_neg_45_deg(image)

    # This algorithm need the initial point, this is where we need to create an automatic method to detect these points.
    result_set = line_tracing(init_point=init_point, grad_x=x_grad, grad_45=grad_45, grad_neg_45=grad_neg_45)
    # result_set = [(i, j) for i in range(0, 150) for j in range(0, 200)]
    result_mask = generate_mask_im(image, result_set)

    result_set_curve_fit = use_curve_fitting(result_set)
    result_mask_ = generate_mask_im(image, result_set_curve_fit)

    im_with_result_set = create_image_with_mask(image_rgb, result_set, color='blue')
    im_with_result_set_curve_fit = create_image_with_mask(image_rgb, result_set_curve_fit)

    titles = ['original image', 'horizontal gradient',
              'derivative at 45 deg', 'derivative at neg 45 deg', 'line detect', 'line after use curve fitting',
              'result image with line detect', 'result image with curve fitting']
    imshow_all(image, x_grad, grad_45, grad_neg_45, result_mask, result_mask_, im_with_result_set,
               im_with_result_set_curve_fit, titles=titles, cmap='gray')

    # imsave('im_7_line_detect__.jpg', im_with_result_set)
    # imsave('im_7_line_detect_use_curve_fit__.jpg', im_with_result_set_curve_fit)
    plt.show()
