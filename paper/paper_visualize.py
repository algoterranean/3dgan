import tensorflow as tf
import numpy as np
#import cv2
import matplotlib as mpl
import argparse
import os
import glob
import hem
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
tableau20 = [(x[0]/255.0, x[1]/255.0, x[2]/255.0) for x in tableau20]

def get_event_files(dir):
    train_dir = os.path.join(dir, 'train')
    validate_dir = os.path.join(dir, 'validate')
    train_event_files = glob.glob('{}/events*'.format(train_dir))
    validate_event_files = glob.glob('{}/events*'.format(validate_dir))
    return train_event_files, validate_event_files

# def plot_events(ax, tag, events, name, dataset, color=0):
#     x_data, y_data = get_scalar(tag, events)
#     line, = ax.plot(x_data, y_data, linewidth=1.0, color=tableau20[color])
#     # plt.text(x_data[-1]+50, y_data[-1], r'{:.2e} ${}_{{{}}}$'.format(y_data[-1], name, dataset), color=tableau20[color])
#     # ax.set_ylim([0, 2*y_data[-1]])
#     return line, name, y_data[-1]

def process_events(event_dirs):
    events = { }
    for k, v in event_dirs.items():
        print('Analyzing {}'.format(k))
        train_event_files, validate_event_files = get_event_files(v)
        train_events = hem.get_all_events(train_event_files)
        validate_events = hem.get_all_events(validate_event_files)
        events[k] = (train_events, validate_events)
    return events

def get_scalar(tag, events):
    vals = hem.get_tag_values('scalar', tag, events)
    x_axis = [x[0] for x in vals]
    y_axis = [x[2] for x in vals]
    return x_axis, y_axis



def plot_stuff(ax, events, ylim=True, tag='metrics_y_hat/linear_rmse_1', color=3, semilogy=False):
    handles = []
    labels = []
    i = color
    for k, v in events.items():
        train_events, validate_events = v

        x_data, y_data = get_scalar(tag, train_events)
        if semilogy:
            line, = ax.semilogy(x_data, y_data, linewidth=1.0, color=tableau20[i])
        else:
            line, = ax.plot(x_data, y_data, linewidth=1.0, color=tableau20[i])
        label = k
        val = y_data[-1]

        # line, label, val = plot_events(ax, tag, train_events, k, 'train', color=i)
        if ylim:
            ax.set_ylim([0, 2 * val])
        handles.append(line)
        labels.append('{}'.format(label))
        i += 2
    return handles, labels

def plot_stuff2(ax, events, ylim=True, tag='metrics_y_hat/linear_rmse_1', color=3, semilogy=False):
    handles = []
    labels = []
    i = color
    n = 0
    for k, v in events.items():
        train_events, validate_events = v

        x_data, y_data = get_scalar(tag, train_events)
        if semilogy:
            line, = ax.semilogy(x_data, y_data, linewidth=1.0, color=tableau20[i])
        else:
            line, = ax.bar(n, y_data[-1], 0.5, color=tableau20[i])
            # line, = ax.plot(x_data, y_data, linewidth=1.0, color=tableau20[i])
        label = k
        val = y_data[-1]

        # line, label, val = plot_events(ax, tag, train_events, k, 'train', color=i)
        if ylim:
            ax.set_ylim([0, 2 * val])
        handles.append(line)
        labels.append('{}'.format(label))
        i += 2
        n += 1
    return handles, labels

def generate_experiment1_charts(baseline_events, mean_adjusted_events, mean_provided_events, out_fn):
    f = plt.figure(figsize=(6,2))
    ax1 = f.add_subplot(1,3,1)
    ax2 = f.add_subplot(1,3,2)
    ax3 = f.add_subplot(1,3,3,sharey=ax2)

    for ax in [ax1, ax2, ax3]:
        ax.yaxis.grid(True, linestyle='dotted')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for s in ['right', 'top', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_color("black")
            line.set_markeredgewidth(0)
        for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
            line.set_markersize(0)
        plt.rc('xtick', direction='out')
        plt.rc('ytick', direction='in')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    if baseline_events is not None:
        handles, labels = plot_stuff(ax1, baseline_events)
    if mean_adjusted_events is not None:
        handles, labels = plot_stuff(ax2, mean_adjusted_events)
    if mean_provided_events is not None:
        handles, labels = plot_stuff(ax3, mean_provided_events)
    ax3.legend(handles, labels, loc="upper right")

    f.text(0.5, 0.05, r'Step', ha='center')
    f.text(0.01, 0.5, r'$\text{RMSE}(y, \hat{y})$', va='center', rotation='vertical')

    ax1.set_xlabel(r'\textbf{(a)} $G(x) = \hat{y}$')
    ax1.xaxis.set_label_position('top')
    ax2.set_xlabel(r'\textbf{(b)} $G(x) = \hat{y} - \bar{y}$')
    ax2.xaxis.set_label_position('top')
    ax3.set_xlabel(r'\textbf{(c)} $G(x, \bar{y}) = \hat{y} - \bar{y}$')
    ax3.xaxis.set_label_position('top')

    plt.tight_layout(pad=2)
    plt.savefig(out_fn)
    plt.show()


def generate_experiment1b_charts(baseline_events, mean_adjusted_events, mean_provided_events, out_fn):
    f = plt.figure(figsize=(6, 2))
    ax1 = f.add_subplot(1, 3, 1)
    ax1b = ax1.twinx()
    ax2 = f.add_subplot(1, 3, 2)
    ax2b = ax2.twinx()
    ax3 = f.add_subplot(1, 3, 3, sharey=ax2)
    ax3b = ax3.twinx()

    for ax in [ax1, ax2, ax3]:
        ax.yaxis.grid(True, linestyle='dotted')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for s in ['right', 'top', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_color("black")
            line.set_markeredgewidth(0)
        for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
            line.set_markersize(0)
        plt.rc('xtick', direction='out')
        plt.rc('ytick', direction='in')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    for ax in [ax1b, ax2b, ax3b]:
        ax.yaxis.grid(False) #, linestyle='dotted')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for s in ['right', 'top', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_color("black")
            line.set_markeredgewidth(0)
        for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
            line.set_markersize(0)
        plt.rc('xtick', direction='out')
        plt.rc('ytick', direction='in')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_right()

    if baseline_events is not None:
        handles, labels = plot_stuff(ax1, baseline_events, tag='loss/loss/discriminator/d_fake')
        handles2, labels2 = plot_stuff(ax1b, baseline_events, tag='metrics_y_hat/linear_rmse_1', color=5)
    if mean_adjusted_events is not None:
        handles, labels = plot_stuff(ax2, mean_adjusted_events, tag='loss/loss/discriminator/d_fake')
        handles2, labels2 = plot_stuff(ax2b, mean_adjusted_events, tag='metrics_y_hat/linear_rmse_1', color=5)
    if mean_provided_events is not None:
        handles, labels = plot_stuff(ax3, mean_provided_events, tag='loss/loss/discriminator/d_fake')
        handles2, labels2 = plot_stuff(ax3b, mean_provided_events, tag='metrics_y_hat/linear_rmse_1', color=5)

    labels = [r'$D$ loss', r'Mean RMSE']
    # print('labels:', labels, labels2)

    ax3.legend(handles + handles2, labels, loc="lower right")

    f.text(0.5, 0.05, r'Step', ha='center')
    f.text(0.01, 0.5, r'$\mathcal{L}_{D(x, \hat{y}}$', va='center', rotation='vertical')
    f.text(0.97, 0.5, r'$\text{RMSE}(y, \hat{y})$', va='center', rotation='vertical')

    ax1.set_xlabel(r'\textbf{(a)} $G(x) = \hat{y}$')
    ax1.xaxis.set_label_position('top')
    ax2.set_xlabel(r'\textbf{(b)} $G(x) = \hat{y} - \bar{y}$')
    ax2.xaxis.set_label_position('top')
    ax3.set_xlabel(r'\textbf{(c)} $G(x, \bar{y}) = \hat{y} - \bar{y}$')
    ax3.xaxis.set_label_position('top')

    plt.tight_layout(pad=2)
    plt.savefig(out_fn)
    plt.show()

def generate_experiment2_charts(rmse_events, variance_events, min_mean_events, out_fn='experiment2.pdf'):
    f = plt.figure(figsize=(6,2))
    ax1 = f.add_subplot(1,3,1)
    ax2 = f.add_subplot(1,3,2)
    ax3 = f.add_subplot(1,3,3)

    for ax in [ax1, ax2, ax3]:
        ax.yaxis.grid(True, linestyle='dotted')
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for s in ['right', 'top', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(5)
            line.set_color("black")
            line.set_markeredgewidth(0)
        for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
            line.set_markersize(0)
        plt.rc('xtick', direction='out')
        plt.rc('ytick', direction='in')
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    # show sampler rmse is comparable to gan w/o noise
    if rmse_events is not None:
        handles, labels = plot_stuff2(ax1, rmse_events, ylim=False)
        ax1.set_ylim([0.02, 0.045])
        ax1_labels = labels
        ax1_handles = handles
        ax1.get_xaxis().set_ticks([])
        # ax1.legend(handles, labels, loc="upper right")

    # show per-image variance for each sampler model, in log-scale
    if variance_events is not None:
        handles, labels = plot_stuff(ax2, variance_events, tag='metrics_y_sampler/g_moments/var', ylim=False, semilogy=True, color=5)

        # ax2.legend(handles, labels, bbox_to_anchor=(1.04,1), borderaxespad=0)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

    if min_mean_events is not None:
        handles = []
        labels = []
        i = 5
        n = 0
        ax3.set_yscale("log")
        # ax3.get_xaxis().set_visible(False)


        for k, v in min_mean_events.items():
            train_events, validate_events = v
            x_data, y_data = get_scalar('metrics_y_sampler/per_image_rmse/mean', train_events)
            x_data2, y_data2 = get_scalar('metrics_y_sampler/per_image_rmse/min', train_events)
            y_data3 = [a - b for a, b in zip(y_data, y_data2)]
            line, = ax3.bar(n, y_data3[-1], 0.5, color=tableau20[i])
            # line, = ax3.semilogy(x_data, y_data3, linewidth=1.0, color=tableau20[i])
            handles.append(line)
            labels.append('{}'.format(k))
            i += 2
            n += 1

            # line, = ax3.semilogy(x_data, y_data, linewidth=1.0, color=tableau20[3])
            # handles.append(line)
            # labels.append('{} {}'.format(k, 'mean'))
            #
            # x_data, y_data = get_scalar('metrics_y_sampler/per_image_rmse/min', train_events)
            # line, = ax3.semilogy(x_data, y_data, linewidth=1.0, color=tableau20[4])
            # handles.append(line)
            # labels.append('{} {}'.format(k, 'min'))
        ax3.get_xaxis().set_ticks([])
        ax3.legend(ax1_handles, ax1_labels, bbox_to_anchor=(1.04, 1), borderaxespad=0)


    # # show per-image mean and min from sampler model
    # if sampler_events is not None:
    #     handles, labels = plot_stuff(ax2, sampler_events, tag='metrics_y_sampler/g_moments/var', ylim=False)
    #     ax2.legend(handles, labels, loc="lower right")
    #     handles, labels = plot_stuff(ax3, mean_min_events, tag='metrics_y_sampler/per_image_rmse/mean')
    #     # ax3.legend(handles, labels, loc="lower right")
    #     handles2, labels2 = plot_stuff(ax3, mean_min_events, tag='metrics_y_sampler/per_image_rmse/min', color=5)
    #     ax3.legend(handles + handles2, labels+labels2, loc="lower right")

    # if mean_adjusted_events is not None:
    #     handles, labels = plot_stuff(ax2, mean_adjusted_events)
    # if mean_provided_events is not None:
    #     handles, labels = plot_stuff(ax3, mean_provided_events)

    f.text(0.5, 0.05, r'Step', ha='center')
    # f.text(0.01, 0.5, r'$\text{RMSE}(y, \hat{y})$', va='center', rotation='vertical')

    ax1.set_xlabel(r'\textbf{RMSE}')
    ax1.xaxis.set_label_position('top')
    ax2.set_xlabel(r'\textbf{Var}')
    ax2.xaxis.set_label_position('top')
    ax3.set_xlabel(r'\textbf{Mean - Min}')
    ax3.xaxis.set_label_position('top')
    # ax3.set_xlabel(r'\textbf{(c)} $G(x, \bar{y}) = \hat{y} - \bar{y}$')
    # ax3.xaxis.set_label_position('top')

    plt.tight_layout(pad=2) #, rect=[0,0,0.75,1])
    plt.savefig(out_fn, bbox_inches="tight")
    plt.show()




plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=[
    r"\usepackage{units}",
    r"\usepackage[utf8x]{inputenc}",
    r"\usepackage{mathtools}",
    r"\usepackage{amssymb}",
    r"\usepackage[lf]{venturis}",
    r"\usepackage[T1]{fontenc}",
    ])
plt.style.use('seaborn-paper')


# # 1. plot CGAN and CNN comparisons
# baseline_dirs = {r'$G_{\ell_2}$': '/mnt/storage/thesis/standalone/baseline',
#                  r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/baseline'}
# mean_adjusted_dirs = {r'$G_{\ell_2}$': '/mnt/storage/thesis/standalone/mean_adjusted',
#                       r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/mean_adjusted'}
# mean_provided_dirs = {r'$G_{\ell_2}$': '/mnt/storage/thesis/standalone/mean_provided',
#                       r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/mean_provided'}
# baseline_events = process_events(baseline_dirs)
# mean_adjusted_events = process_events(mean_adjusted_dirs)
# mean_provided_events = process_events(mean_provided_dirs)
#
# generate_experiment1_charts(baseline_events, mean_adjusted_events, mean_provided_events, 'experiment1.pdf')
#
#
# baseline_dirs = {r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/baseline'}
# mean_adjusted_dirs = {r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/mean_adjusted'}
# mean_provided_dirs = {r'$G_{\text{cGAN}}$': '/mnt/storage/thesis/cgan/mean_provided'}
# baseline_events = process_events(baseline_dirs)
# mean_adjusted_events = process_events(mean_adjusted_dirs)
# mean_provided_events = process_events(mean_provided_dirs)
#
# generate_experiment1b_charts(baseline_events, mean_adjusted_events, mean_provided_events, 'experiment1b.pdf')



# 2. plot CGAN_delta and CGAN_noise comparisons
rmse_dirs = {r'$none$': '/mnt/storage/thesis/cgan/mean_adjusted',
             r'$x$': '/mnt/storage/thesis/sampler/baseline_x',
             r'$e_1$': '/mnt/storage/thesis/sampler/baseline_e1',
             r'$e_2$': '/mnt/storage/thesis/sampler/baseline_e2',
             r'$e_3$': '/mnt/storage/thesis/sampler/baseline_e3',
             r'$e_4$': '/mnt/storage/thesis/sampler/baseline_e4-512',
             r'$d_2$': '/mnt/storage/thesis/sampler/baseline_d2',
             r'$d_3$': '/mnt/storage/thesis/sampler/baseline_d3',
             r'$d_4$': '/mnt/storage/thesis/sampler/baseline_d4'}

variance_dirs = {r'$x$': '/mnt/storage/thesis/sampler/baseline_x',
                 r'$e_1$': '/mnt/storage/thesis/sampler/baseline_e1',
                 r'$e_2$': '/mnt/storage/thesis/sampler/baseline_e2',
                 r'$e_3$': '/mnt/storage/thesis/sampler/baseline_e3',
                 r'$e_4$': '/mnt/storage/thesis/sampler/baseline_e4-512',
                 r'$d_2$': '/mnt/storage/thesis/sampler/baseline_d2',
                 r'$d_3$': '/mnt/storage/thesis/sampler/baseline_d3',
                 r'$d_4$': '/mnt/storage/thesis/sampler/baseline_d4'}

min_mean_dirs = variance_dirs
rmse_events = process_events(rmse_dirs)
variance_events = process_events(variance_dirs)
min_mean_events = process_events(min_mean_dirs)

generate_experiment2_charts(rmse_events, variance_events, min_mean_events, 'experiment2.pdf')




