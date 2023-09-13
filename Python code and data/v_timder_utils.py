import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime
from utils import read_tif
from config import paths, color_scheme


def draw_now(widgets_dict, graphic_handles, data, selection_mode):
    """
    This draws all the graphics
    Args:
        widgets_dict (dict): contains all the instances of the widgets
        graphic_handles (dict): contains all the instances of the graphic handles (fig, axes, images etc.)
        data (dict): contains all the data (df + crops of varying sizes)
        selection_mode (str): a constant initialized in the notebook, either "binary" or "mix"
    """
    fig = graphic_handles["fig"]
    ax = graphic_handles["ax"]
    I_arr = graphic_handles["I_arr"]
    plot_arr = graphic_handles["plot_arr"]

    n = widgets_dict["current_index"].value

    if selection_mode.lower() == "binary":
        mir_type = data["df"]["mir_type"].values[n]
    elif selection_mode.lower() == "mix":
        mir_type = widgets_dict["select_mir_type_widget"].value

    min_cmap = widgets_dict["min_slider"].value
    max_cmap = widgets_dict["max_slider"].value

    if widgets_dict["large_context_widget"].value == "Normal":
        ref_crops_str = "reference_crops"
        noisy_crops_str = "crops_noisy"
        denoised_crops_str = "crops_denoised"
    elif widgets_dict["large_context_widget"].value == "Large context":
        ref_crops_str = "large_reference_crops"
        noisy_crops_str = "crops_noisy_large_context"
        denoised_crops_str = "crops_denoised_large_context"
    else:
        ref_crops_str = "extra_large_reference_crops"
        noisy_crops_str = "crops_noisy_extra_large_context"
        denoised_crops_str = "crops_denoised_extra_large_context"

    I_arr[0].set_data(data[ref_crops_str][mir_type])  # reference crops
    I_arr[1].set_data(data[noisy_crops_str][n])  # noisy crops
    I_arr[2].set_data(data[denoised_crops_str][n])  # denoised crops
    ax[3].cla()
    I_arr[3] = sns.histplot(data["crops_noisy_extra_large_context"][n].flatten(), color=[0.5, 0.5, 0.5],
                            ax=ax[3])  # small histogram
    xdata, ydata = mir_assistance_points(mir_type, widgets_dict["large_context_widget"].value)
    # assistance points
    for j in range(3):
        plot_arr[j].set_xdata(xdata)
        plot_arr[j].set_ydata(ydata)
        plot_arr[j].set_color(color_scheme[mir_type])
    # assistance line on histogram
    plot_arr[3] = ax[3].plot([min_cmap, max_cmap], ax[3].get_ylim(), "k", linewidth=0.5)[0]
    # FOV
    plot_arr[4].set_xdata(data["df"]["x"].values[n])
    plot_arr[4].set_ydata(data["df"]["y"].values[n])
    plot_arr[4].set_color(color_scheme[mir_type])

    # title
    fig.suptitle(f"Crop no. {n}", color=color_scheme[mir_type])

    # apply min/max (brightness/contrast)
    if min_cmap >= max_cmap:
        min_cmap = 0
    for i in range(1, 3):
        I_arr[i].set_clim([min_cmap, max_cmap])

    # colored title
    ax[0].set_title(ax[0].get_title(), color=color_scheme[mir_type])

    # display denoised according to checkbox widget
    ax[2].set_visible(widgets_dict["display_denoised_widget"].value)


def mir_assistance_points(mir_type, context):
    """
    This encompasses the specific details about the miRs and the optics to generate convenient assistance points
    Args:
        mir_type (int): 0 1 or 2
        context (str): "Normal", "Large context" or "XL context"
    Returns:
        xdata, ydata (floats): x,y coordinates of the assistance points

    """
    xdata = [4.5, 4.5]
    if context == 'Normal':
        if mir_type == 0:
            ydata = [5.5, 17]
        elif mir_type == 1:
            ydata = [5.5, 9.5]
        elif mir_type == 2:
            ydata = [5.5, 12]
    elif context == 'Large context':
        if mir_type == 0:
            ydata = [5.5, 11.25]
        elif mir_type == 1:
            ydata = [5.5, 7.5]
        elif mir_type == 2:
            ydata = [5.5, 8.75]
    else:
        if mir_type == 0:
            ydata = [5.5, 6.65]
        elif mir_type == 1:
            ydata = [5.5, 5.9]
        elif mir_type == 2:
            ydata = [5.5, 6.15]
    return xdata, ydata


## This is from another project for later implementation for UI in jupyter notebooks
def notebook_setup_widgets(widgets, graphic_handles, data, selection_mode):
    """
    This creates all the notebook widgets for V_TIMDER and returns them as a dict.
    This also assigns the widgets what to do when they are pressed.
    Finally this generates the important "local_draw_now" function which in essence is called after every click to
    refresh the display
    Args:
        widgets (module): instance of the imported widgets module
        graphic_handles (dict): contains all the instances of the graphic handles (fig, axes, images etc.)
        data (dict): contains all the data (df + crops of varying sizes)
        selection_mode (str): a constant initialized in the notebook, either "binary" or "mix"
    Returns:
        widgets_dict (dict): dict with all the instances of the widgets
        local_draw_now (function): an instance of the function draw_now() with all the widget/graphic/data instances
        filled in already.
    """

    #Create all the widgets
    select_mir_type_widget = widgets.RadioButtons(options=[0, 1, 2], description="miR type:",
                                                  value=0, disabled=False)
    large_context_widget = widgets.RadioButtons(options=['Normal', 'Large context', 'XL context'],
                                                value='Normal', disabled=False)
    display_denoised_widget = widgets.Checkbox(value=True, description="Display denoised?", disabled=False,
                                               indent=False)

    min_slider = widgets.IntSlider(value=800, min=100, max=1999, step=1, description='Minimum', disabled=False,
                                   continuous_update=False, orientation='horizontal', readout=True, )
    max_slider = widgets.IntSlider(value=1000, min=101, max=2000, step=1, description='Maximum', disabled=False,
                                   continuous_update=False, orientation='horizontal', readout=True, )
    current_index = widgets.BoundedIntText(min=0, max=10 ** 7, step=1, description='Current crop #:',
                                           style={'description_width': 'initial'}, disabled=False)
    positive_button = widgets.Button(
        description='This is a miR!',
        disabled=False,
        button_style='Success',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Positive',
        icon='fa-check'  # (FontAwesome names without the `fa-` prefix)
    )
    negative_button = widgets.Button(
        description='This is NOT a miR!',
        disabled=False,
        button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Negative',
        icon='close'  # (FontAwesome names without the `fa-` prefix)
    )
    #pack everything in a dict
    widgets_dict = {
        "select_mir_type_widget": select_mir_type_widget,
        "large_context_widget": large_context_widget,
        "display_denoised_widget": display_denoised_widget,
        "min_slider": min_slider,
        "max_slider": max_slider,
        "positive_button": positive_button,
        "negative_button": negative_button,
        "current_index": current_index
    }

    # This is a local instance of the draw_now() function with all the instances of widgets/graphics/data filled in
    def local_draw_now(change):
        """
        This is the function that is to be called every time the graphics are to be refreshed
        Args:
            change (None): has no effect but an argument is needed apparently for this setup
        """
        draw_now(widgets_dict, graphic_handles, data, selection_mode)

    #Make each widget react according with local_draw_now
    select_mir_type_widget.observe(local_draw_now, names='value')
    large_context_widget.observe(local_draw_now, names='value')
    display_denoised_widget.observe(local_draw_now, names='value')
    min_slider.observe(local_draw_now, names='value')
    max_slider.observe(local_draw_now, names='value')
    current_index.observe(local_draw_now, names='value')

    #Define the reactions for the positive and negative buttons
    def for_positive_button(self):
        """What to do when the positive green button is pressed"""
        n = widgets_dict["current_index"].value
        data["df"].loc[n, "is_really_mir"] = 1
        if n < len(data["df"]) - 1:
            widgets_dict["current_index"].value = n + 1

        if selection_mode.lower() == "mix":
            data["df"].loc[n, "selected_mir_type"] = widgets_dict["select_mir_type_widget"].value

        if (n % 10) == 0 or n == (len(data["df"]) - 1):
            data["df"].to_csv(os.path.join(paths["v_timder_tables"], f"cur_df_{now_string()}.csv"))
        local_draw_now(self)

    def for_negative_button(self):
        """What to do when the negative red button is pressed"""
        n = widgets_dict["current_index"].value
        data["df"].loc[n, "is_really_mir"] = 0
        if n < len(data["df"]) - 1:
            widgets_dict["current_index"].value = n + 1

        if selection_mode.lower() == "mix":
            data["df"].loc[n, "selected_mir_type"] = widgets_dict["select_mir_type_widget"].value

        if (n % 10) == 0 or n == (len(data["df"]) - 1):
            data["df"].to_csv(os.path.join(paths["v_timder_tables"], f"cur_df_{now_string()}.csv"))
        local_draw_now(self)

    widgets_dict["positive_button"].on_click(for_positive_button)
    widgets_dict["negative_button"].on_click(for_negative_button)

    #now all the widgets are configured with the proper reactions to clicks and drags
    return widgets_dict, local_draw_now


def display_widgets(widgets, widgets_dict, selection_mode):
    """
    This displays the widgets. "display" is a function of ipython in jupyter notebooks
    Args:
        widgets (module): instance of the imported widgets module
        widgets_dict (dict): dict with all the instances of the widgets
        selection_mode (str): a constant initialized in the notebook, either "binary" or "mix"
    """

    # display everything using vertical and horizontal boxes
    box1 = widgets.VBox([widgets_dict["large_context_widget"], widgets_dict["display_denoised_widget"]])

    box2 = widgets.VBox([widgets_dict["min_slider"], widgets_dict["max_slider"]])
    if selection_mode.lower() == "mix":
        box2 = widgets.HBox([box2, widgets_dict["select_mir_type_widget"]])

    box3 = widgets.HBox([widgets_dict["negative_button"], widgets_dict["positive_button"]])
    box4 = widgets.HBox([box2, box1])
    display(widgets.HBox([box3, box4]))
    display(widgets_dict["current_index"])


def generate_graphic_handles(data):
    """
    This generates the graphic handles (fig, ax etc.). From now on they are just modified, not generated again.
    Args:
        data (dict): contains all the data (df + crops of varying sizes)
    Returns:
        graphic_handles (dict): contains all the graphic handles (fig, ax, images, plots)
    """
    reference_crops, crops_noisy, crops_denoised = data["reference_crops"], data["crops_noisy"], data["crops_denoised"]
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[:, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 3])
    all_ax = np.asarray([ax1, ax2, ax3, ax4, ax5])

    titles = ["Reference", "Raw", "Denoised", "B&C", "FOV"]
    I_arr = np.empty(5, dtype=object)
    plot_arr = np.empty(5, dtype=object)
    for i, ax in enumerate(all_ax):
        plt.axes(ax)
        if i == 0:
            I_arr[i] = plt.imshow(reference_crops[0, ...], cmap="gray")
        elif i == 1:
            I_arr[i] = plt.imshow(crops_noisy[0, ...], cmap="gray")
        elif i == 2:
            I_arr[i] = plt.imshow(crops_denoised[0, ...], cmap="gray")
        elif i == 3:
            I_arr[i] = sns.histplot(crops_noisy[0, ...].flatten(), color=[0.5, 0.5, 0.5])
            ax.get_yaxis().set_visible(False)
        elif i == 4:
            I_arr[i] = plt.imshow(np.zeros([1024, 1024], dtype=bool), cmap="gray")
        ax.set_title(titles[i])
        if i < 3:
            plt.axis('off')
            plot_arr[i] = plt.plot([4.5, 4.5], [3, 15], 'or', alpha=0.6, markerfacecolor='none')[0]
        elif i == 3:
            plot_arr[i] = plt.plot([0, 0.5], ax.get_ylim(), "k")[0]
        elif i == 4:
            plot_arr[i] = plt.plot([500], [500], "w.", markersize=10)[0]
            plt.axis('off')

    graphic_handles = {
        "fig": fig,
        "ax": all_ax,
        "I_arr": I_arr,
        "plot_arr": plot_arr
    }
    return graphic_handles


def now_string():
    """Returns a string of current date/time"""
    return datetime.now().strftime('%y_%m_%d__%H_%M_%S')


def load_data(filename, ref_crops_name):
    """
    This loads all the data from the V_TIMDER folders, in the format exported from the "Image_Processing" notebook
    Args:
        filename (str): name of the file according to which the data is found
        ref_crops_name (str): name of the file with the reference crops
    Returns:
        data (dict): contains all the data (df + crops of varying sizes)
    """
    reference_crops = read_tif(os.path.join(paths["v_timder_reference"], "Normal", ref_crops_name))
    crops_noisy = read_tif(os.path.join(paths["v_timder_noisy"], "Normal", filename))
    crops_denoised = read_tif(os.path.join(paths["v_timder_denoised"], "Normal", filename))

    reference_crops_large = read_tif(os.path.join(paths["v_timder_reference"], "Large", ref_crops_name))
    crops_noisy_large = read_tif(os.path.join(paths["v_timder_noisy"], "Large", filename))
    crops_denoised_large = read_tif(os.path.join(paths["v_timder_denoised"], "Large", filename))

    reference_crops_XL = read_tif(os.path.join(paths["v_timder_reference"], "XL", ref_crops_name))
    crops_noisy_XL = read_tif(os.path.join(paths["v_timder_noisy"], "XL", filename))
    crops_denoised_XL = read_tif(os.path.join(paths["v_timder_denoised"], "XL", filename))

    data = {
        "reference_crops": reference_crops,
        "crops_noisy": crops_noisy,
        "crops_denoised": crops_denoised,
        "large_reference_crops": reference_crops_large,
        "crops_noisy_large_context": crops_noisy_large,
        "crops_denoised_large_context": crops_denoised_large,
        "extra_large_reference_crops": reference_crops_XL,
        "crops_noisy_extra_large_context": crops_noisy_XL,
        "crops_denoised_extra_large_context": crops_denoised_XL,
    }
    return data
