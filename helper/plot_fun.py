# encoding: utf-8

import helper.misc as hmisc
from mpl_toolkits.axes_grid1 import axes_grid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


def simple_plot_dict(input_dict):
    # Just seeing if this works or not....
    # input_dict = loss_dict
    NUM_COLORS = len(input_dict)

    cm = plt.get_cmap('Reds')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    for k in sorted(input_dict.keys()):
        ax.plot(input_dict[k], label=k)
    plt.legend(loc='upper left')


def plot_multi_lines(data, cmap_col='Reds', style='.'):
    """
    Example...

    import numpy as np
    import matplotlib.pyplot as plt

    N = 100
    n_line = 10
    x = np.linspace(0, 2 * np.pi, N)
    noise = np.random.rand(N)
    y = np.sin(x)
    y_set = [y + noise ** np.random.rand() for i in range(n_line)]
    y_set = np.vstack(y_set).T
    y_set.shape

    plot_multi_lines(y_set)
    plt.show()
    :param data:
    :param cmap_col:
    :return:
    """
    num_lines = data.shape[1]
    cm = plt.get_cmap(cmap_col)
    color_list = [cm(1. * i / num_lines) for i in range(num_lines)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', color_list)
    for i in range(num_lines):
        ax.plot(data[:, i], marker=style)


def plot_multi_points(data, cmap_col='Reds'):
    """

    :param data: nd-array of size n_data x 2
    :param cmap_col: 'Reds'
    :param legend: binary, yes or no legend
    :param alpha: the.. transparancy
    :return:
    """

    num_lines = data.shape[0]
    cm = plt.get_cmap(cmap_col)
    color_list = [cm(1. * i / num_lines) for i in range(num_lines)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', color_list)
    for i in range(num_lines):
        ax.scatter(data[i, 0], data[i, 1], label=i)


def plot_surf(data):
    # Expect input: (x,y)
    from mpl_toolkits.mplot3d import Axes3D

    # Set up grid and test data
    nx, ny = data.shape
    x = range(nx)
    y = range(ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X.T, Y.T, data)
    # We could use plot_surface here.. or contourf with offset to do something fancy


def subplot_results(first_image, second_image, fignum=None):
    """
    Plot the results...
    :param model_predict:
    :param ground_truth:
    :return:
    """
    plt.figure(fignum)
    plt.subplot(1, 2, 1)
    plt.imshow(first_image)
    plt.title('Prediction')
    plt.subplot(1, 2, 2)
    plt.imshow(second_image)
    plt.title('Ground truth')


def subplot_layers(input_ndarray, batch_nr=0, std_color='blue', title=''):
    """
    Plot the intermediate layers of a model...
    Or just any ndarray...

    :param input_ndarray: (format: [batch, x, y, channels]
    :return:
    """
    # Create figure
    fig_num = len(plt.get_fignums()) + 1
    
    
    # Determine size of subplots
    n_size = np.min([input_ndarray.shape[-1], 100])
    n_xsize = int(np.ceil(n_size ** 0.5))

    # Create subplots
    fig_sub, ax_sub = plt.subplots(n_xsize, n_xsize, num=fig_num)
    fig_sub.suptitle(title)
    if isinstance(ax_sub, np.ndarray):
        ax_list = ax_sub.ravel()
    else:
        ax_list = [ax_sub]

    # Plot relevant data
    for i, i_ax in enumerate(ax_list[:n_size]):
        i_ax.imshow(input_ndarray[batch_nr, :, :, i])
    
    # Plot irrelevant data
    for i, i_ax in enumerate(ax_list[n_size:]):    
        i_ax.set_facecolor(std_color)
        

def subplot_interactive(input_array, i_fig=1):
    """
    :param input_array:
    :return:
    """
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)

#     plt.figure(i_fig)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    sel_chan = 4
    n_chan = input_array.shape[-1]
    derp = plt.imshow(input_array[:, :, sel_chan], cmap='gray')

    axfreq = plt.axes([0.25, 0.1, 0.40, 0.03])
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])

    axbutton_left = plt.axes([0.10, 0.1, 0.10, 0.03])
    #axbutton_right = plt.axes([0.70, 0.1, 0.10, 0.03])

    sSlice = Slider(axfreq, 'Slice', 0, n_chan, valinit=sel_chan, valfmt='%0.0f')
    button_new = Button(resetax, 'Reset', hovercolor='0.975')
    button_left = Button(axbutton_left, 'Left', hovercolor='0.975')
    #button_right = Button(axbutton_right, 'Right', hovercolor='0.975')

    def update(val):
        derp.set_data(input_array[:, :, int(sSlice.val)])

    def reset(event):
        print('are we in')
        sSlice.reset()

    button_new.on_clicked(reset)


    def left(derpederp):
        newwindow = max(int(sSlice.val) - 1, 0)
        print(newwindow, int(sSlice.val))
        derp.set_data(input_array[:, :, newwindow])

    def right(derpederp):
        newwindow = min(int(sSlice.val) + 1, n_chan)
        print(newwindow, int(sSlice.val))
        derp.set_data(input_array[:, :, newwindow])

    button_left.on_clicked(left)
    sSlice.on_changed(update)


def redraw_fn(f, axes):
    # Used to draw a video...
    global _imagelist
    img = _imagelist[:, :, f]
    if not redraw_fn.initialized:
        redraw_fn.im = axes.imshow(img, animated=True, cmap='gray')
        redraw_fn.initialized = True
    else:
        redraw_fn.im.set_array(img)

# redraw_fn.initialized = False
# videofig(len(_imagelist), redraw_fn, play_fps=60)


def plot_3d_list(image_list, **kwargs):
    # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..
    figsize = kwargs.get('figsize')
    fignum = kwargs.get('fignum')
    dpi = kwargs.get('dpi')

    title_string = kwargs.get('title', "")
    sub_title = kwargs.get('subtitle', None)
    cbar_ind = kwargs.get('cbar', False)

    vmin = kwargs.get('vmin', None)
    ax_off = kwargs.get('ax_off', False)
    augm_ind = kwargs.get('augm', None)
    aspect_mode = kwargs.get('aspect', 'equal')

    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.1)
    debug = kwargs.get('debug', False)

    f = plt.figure(fignum, figsize, dpi)
    f.suptitle(title_string)

    # Only when we have an numpy array
    if isinstance(image_list, np.ndarray):
        # With just two dimensions..
        if image_list.ndim == 2:
            # Add one..
            image_list = image_list[np.newaxis]

    n_rows = len(image_list)
    gs0 = gridspec.GridSpec(n_rows, 1, figure=f)
    gs0.update(wspace=wspace, hspace=hspace)  # set the spacing between axes.

    print('amount of rows..', n_rows)
    for i, i_gs in enumerate(gs0):
        temp_img = image_list[i]

        if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
            if temp_img.ndim == 4:
                n_sub_col = temp_img.shape[0]
                n_sub_row = temp_img.shape[1]
                temp_img = temp_img.reshape((n_sub_col * n_sub_row, ) + temp_img.shape[2:])
            elif temp_img.ndim == 3:
                n_sub_col = temp_img.shape[0]
                if n_sub_col > 8:
                    n_sub_col, n_sub_row = hmisc.get_square(n_sub_col)
                else:
                    n_sub_row = 1
            else:
                temp_img = temp_img[np.newaxis]
                n_sub_col = 1
                n_sub_row = 1
        else:
            n_sub_col = len(temp_img)
            n_sub_row = 1

        # If we want to specifcy the vmin per list item.. we can do that here..
        if isinstance(vmin, list):
            sel_vmin = vmin[i]
        else:
            sel_vmin = vmin

        for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col)):

            ax = f.add_subplot(ii_gs)
            if augm_ind:
                plot_img = eval('{fun}({var})'.format(fun=augm_ind, var=str('temp_img[j]')))
                if 'angle' in augm_ind:
                    sel_vmin = (-np.pi, np.pi)
            else:
                plot_img = temp_img[j]

            if debug:
                print(f'shape {i} - {temp_img.shape}', end=' \t|\t')
                print(f'row/col {n_sub_row} - {n_sub_col}', end=' \t|\t')
                print(f'shape {j} - {plot_img.shape}', end=' \t|\n')

            map_ax = ax.imshow(plot_img, vmin=sel_vmin, aspect=aspect_mode, cmap='gray')

            if cbar_ind:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                temp_cbar = plt.colorbar(map_ax, cax=cax)
                if sel_vmin is None:
                    vmin_temp = [np.min(plot_img), np.max(plot_img)]
                    map_ax.set_clim(vmin_temp)
                    temp_cbar.set_ticks(vmin_temp)
                else:
                    map_ax.set_clim(vmin)
                    temp_cbar.set_ticks(vmin)

            if sub_title is not None:
                ax.set_title(sub_title[i][j])
            if ax_off:
                ax.set_axis_off()

    return f
