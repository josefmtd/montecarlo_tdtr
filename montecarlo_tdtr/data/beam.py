import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, std, mu):
    """ Gaussian distribution f_g (x)
    Args:
        x (double): x
        std (double): standard deviation
        mu (double): mean
    """
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std) ** 2)

def inverse_gaussian(max_val, std, mu):
    """ Inverse gaussian function f_g^(-1)
    """
    c = 1 / (std * np.sqrt(2 * np.pi))
    return mu + std * np.sqrt(2 * (np.log(c) - np.log(max_val)))


def full_width_half_maximum(max_val, std, mu, max_ind, length):
    '''
    '''
    indices = np.arange(0, length * 10000, 1) / 10000
    values = gaussian(indices, std, mu)

    max_ind_new = max_ind * 10000

    first_values = values[:max_ind_new]
    last_values = values[max_ind_new:]

    first_val = indices[np.argmin(np.abs(first_values - (max_val / 2)))]
    last_val = indices[np.argmin(np.abs(last_values - (max_val / 2))) + len(first_values)]

    return last_val - first_val


def ellipticity(a, c):
    """ Calculate ellipticity
    https: // mathworld.wolfram.com / Ellipticity.html
    Args:
        a (numeric) : equatorial radius
        c (numeric) : polar radius
    Returns:
        (numeric, string)
        numeric : ellipticity
        string : "oblate" or "prolate"
    """
    if a >= c:
        return np.sqrt((a ** 2 - c ** 2) / a ** 2), "oblate"
    else:
        return np.sqrt((c ** 2 - a ** 2) / a ** 2), "prolate"


def get_beam_radius(path_to_file, center_in = None, interactive = False, 
                    plot = True, verbose = True):
    # Read image
    image = read_beam_image(path_to_file)
    
    # Apply a Gaussian blur with kernel size of 10 to identify peak
    blur = cv2.blur(image, ksize = (10, 10))

    if center_in is not None:
        center = center_in
    else:
        center = np.unravel_index(np.argmax(blur, axis = None), blur.shape)
        
    if verbose:
        print(center)
    
    # Assign new center by 
    if interactive:
        temp = blur.copy()
        temp = cv2.circle(
            img         = temp, 
            center      = (center[1], center[0]), 
            radius      = 1, 
            color       = (100, 0, 0), 
            thickness   = -1
        )

        fig1, ax = plt.subplots(1, 1)
        ax.imshow(temp)
        ax.set_xlim([center[1] - 100, center[1] + 100])
        ax.set_ylim([center[0] - 100, center[0] + 100])
        plt.show()

        new_center = input("Give new index separated by ' '(* to keep old index): ")

        if new_center != "*":
            center = list(map(int, new_center.split(" ")))[::-1]

            temp = blur.copy()
            temp = cv2.circle(temp, (center[1], center[0]), 5, (100, 0, 0), -1)

            plt.imshow(temp)
            plt.xlim([center[1] - 100, center[1] + 100])
            plt.ylim([center[0] - 100, center[0] + 100])
            plt.show()

    # Filter only non-zeros data for Gaussian fitting
    i_x = 1
    i_y = 1

    while (image[center[0], center[1]-i_y:center[1]+i_y+1][0] != 0) or \
        (image[center[0], center[1]-i_y:center[1]+i_y+1][-1] != 0):
        i_y += 1

    while (image[center[0]-i_x:center[0]+i_x+1, center[1]][0] != 0) or \
        (image[center[0]-i_x:center[0]+i_x+1, center[1]][-1] != 0):
        i_x += 1

    # Derive the Gaussian-fit initial values from pixel values
    sum_x = np.sum(image[center[0]-i_x:center[0]+i_x+1, center[1]])
    std_x = np.std(image[center[0]-i_x:center[0]+i_x+1, center[1]])
    mean_x = center[0]

    sum_y = np.sum(image[center[0], center[1]-i_y:center[1]+i_y+1])
    std_y = np.std(image[center[0], center[1]-i_y:center[1]+i_y+1])
    mean_y = center[1]

    x = np.arange(0, image.shape[0], 1)
    y = np.arange(0, image.shape[1], 1)

    # Gaussian fitting
    param_optimised_x, param_covariance_matrix_x = curve_fit(
        gaussian, x, image[:, center[1]]/sum_x, p0=[std_x, mean_x], maxfev=5000
    )

    param_optimised_y, param_covariance_matrix_y = curve_fit(
        gaussian, y, image[center[0]]/sum_y, p0=[std_y, mean_y], maxfev=5000
    )

    gauss_x = gaussian(x, param_optimised_x[0], param_optimised_x[1])
    gauss_y = gaussian(y, param_optimised_y[0], param_optimised_y[1])

    # Plot the data against the Gaussian fit
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, 
                                       figsize = (9.6, 4.8))
        ax1.plot(image[:, center[1]]/sum_x, label = 'Measured')
        ax1.plot(x, gauss_x, label = 'Gaussian')
        ax1.legend()
        ax1.grid('both')

        x_min = param_optimised_x[1] - 5 * param_optimised_x[0]
        x_max = param_optimised_x[1] + 5 * param_optimised_x[0]
        ax1.set_xlim(x_min, x_max)

        ax2.plot(image[center[0]] / sum_y, label = 'Measured')
        ax2.plot(y, gauss_y, label = 'Gaussian')
        ax2.legend()
        ax2.grid('both')

        y_min = param_optimised_y[1] - 5 * param_optimised_y[0]
        y_max = param_optimised_y[1] + 5 * param_optimised_y[0]
        ax2.set_xlim(y_min, y_max)

        ax1.set_ylabel('Intensity [a.u.]')
        ax1.set_xlabel('X position [px]')
        ax2.set_xlabel('Y position [px]')

        fig.show()
        plt.show()

    # Use the FWHM from the Gaussian fit
    width_x = full_width_half_maximum(
        np.amax(gauss_x), param_optimised_x[0], param_optimised_x[1],
        np.argmax(gauss_x), len(gauss_x)
    )

    width_y = full_width_half_maximum(
        np.amax(gauss_y), param_optimised_y[0], param_optimised_y[1],
        np.argmax(gauss_y), len(gauss_y)
    )

    # Transform FWHM to 1/e2 radii
    e, type_e = ellipticity(width_x, width_y)
    ee2_radius = lambda w: np.sqrt(2) * w / np.sqrt(np.log(2)) * 0.55 / 2

    x_um = ee2_radius(width_x)
    y_um = ee2_radius(width_y)
    r_um = (x_um + y_um) / 2

    if plot and verbose:
        temp = image.copy()
        temp = cv2.circle(
            img         = temp, 
            center      = (center[1], center[0]), 
            radius      = 2, 
            color       = (100, 0, 0), 
            thickness   = -1
        )

        if type_e == 'oblate':
            temp = cv2.ellipse(
                img = temp, 
                center = (center[1], center[0]),
                axes = (int(width_x/2), int(width_y/2)),
                angle = 0, 
                startAngle = 0, 
                endAngle = 360,
                color = (100, 0, 0), 
                thickness = 1)
        else:
            temp = cv2.ellipse(
                img = temp, 
                center = (center[1], center[0]),
                axes = (int(width_y/2), int(width_x/2)),
                angle = 90, 
                startAngle = 0, 
                endAngle = 360,
                color = (100, 0, 0), 
                thickness = 1)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(temp)
        ax.set_xlim([center[1] - 0.75 * width_x, 
                     center[1] + 0.75 * width_x])
        ax.set_ylim([center[0] - 0.75 * width_y,
                     center[0] + 0.75 * width_y])
        plt.show()
    
    if verbose:
        print('X-axis 1/e2 radius in um: %.1f' % x_um)
        print('Y-axis 1/e2 radius in um: %.1f' % y_um)
        print('Ellipticity: %.3f' % e, "(" + type_e + ")")
        print("\nAverage 1/e2 radius (X radius + Y radius / 2): %.1f" % r_um)

    return (center, (width_x, width_y))

def draw_ellipse_beam(image, center, width, type_e):
    if type_e == 'oblate':
        image = cv2.ellipse(
            img = image, 
            center = (center[1], center[0]),
            axes = (int(width[0]/2), int(width[1]/2)),
            angle = 0, 
            startAngle = 0, 
            endAngle = 360,
            color = (100, 0, 0), 
            thickness = 1)
    else:
        image = cv2.ellipse(
            img = image, 
            center = (center[1], center[0]),
            axes = (int(width[1]/2), int(width[0]/2)),
            angle = 90, 
            startAngle = 0, 
            endAngle = 360,
            color = (100, 0, 0), 
            thickness = 1)
    return image

def read_beam_image(path_to_file):
    # Read image as numpy array and set background to zero
    image = np.array(cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE))
    image[image <= np.ceil(np.mean(image))] = 0

    return image

def characterize_beam(red_beam, blue_beam, center = None, use_center = True,
                      verbose = False, plot = False):
    # Read image
    red  = read_beam_image(red_beam)
    blue = read_beam_image(blue_beam)

    pump_center, pump_widths = get_beam_radius(blue_beam,
        center_in = center, verbose = verbose, plot = plot)
    probe_center, probe_widths = get_beam_radius(red_beam,
        center_in = center, verbose = verbose, plot = plot)

    blue_e, blue_type_e = ellipticity(pump_widths[0], pump_widths[1])
    red_e, red_type_e = ellipticity(probe_widths[0], probe_widths[1])

    red = draw_ellipse_beam(red, probe_center, probe_widths, red_type_e)
    red = draw_ellipse_beam(red, pump_center, pump_widths, blue_type_e)
    blue = draw_ellipse_beam(blue, probe_center, probe_widths, red_type_e)
    blue = draw_ellipse_beam(blue, pump_center, pump_widths, blue_type_e)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (9.6, 4.8))
    ax1.imshow(blue)
    ax1.set_xlim([probe_center[1] - pump_widths[1], 
        probe_center[1] + pump_widths[1]])
    ax1.set_ylim([probe_center[0] - pump_widths[0],
        probe_center[0] + pump_widths[0]])
    ax2.imshow(red)
    ax2.set_xlim([probe_center[1] - pump_widths[1], 
        probe_center[1] + pump_widths[1]])
    ax2.set_ylim([probe_center[0] - pump_widths[0],
        probe_center[0] + pump_widths[0]])
    fig.show()

    ee2_radius = lambda w: np.sqrt(2) * w / np.sqrt(np.log(2)) * 0.55 / 2
    r_pump  = np.mean(list(map(ee2_radius, pump_widths)))  * 1e-6
    r_probe = np.mean(list(map(ee2_radius, probe_widths))) * 1e-6

    return (r_pump, r_probe)