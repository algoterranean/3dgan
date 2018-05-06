import matplotlib as mpl
# mpl.use('pgf')
# print('MPL BACKENDS',mpl.rcsetup.all_backends)
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import glob
import sys
from operator import itemgetter

import hem




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    train_dir = os.path.join(args.dir, 'train')
    validate_dir = os.path.join(args.dir, 'validate')
    train_event_files = glob.glob('{}/events*'.format(train_dir))
    validate_event_files = glob.glob('{}/events*'.format(validate_dir))

    train_events = hem.get_all_events(train_event_files)
    validate_events = hem.get_all_events(validate_event_files)

    for k, v in train_events.items():
        print('Found {} {} events.'.format(len(v), k))


    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=[
        r"\usepackage{units}",
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage{mathtools}",
        r"\usepackage{amssymb}",
        r"\usepackage[lf]{venturis}",
        r"\usepackage[T1]{fontenc}",
        ])
    plt.style.use('ggplot')
    # plt.style.use('seaborn-paper')
    # plt.style.use('fivethirtyeight')
    # plt.style.use('seaborn')
    # mpl.verbose.level = 'debug-annoying'
    # plt.figure(figsize=(8, 6))

    def get_scalar(tag, events):
        vals = hem.get_tag_values('scalar', tag, events)
        x_axis = [x[0] for x in vals]
        y_axis = [x[2] for x in vals]
        return x_axis, y_axis


    fig = plt.figure(figsize=(8,6))
    # ax = plt.subplot(111, aspect='equal')

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,wspace=0, hspace=0)

    # fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax = plt.gca()

    ax.grid(b=True, which='both', axis='y', color='black', linestyle='--', linewidth=0.5, alpha=0.3, clip_box=mpl.transforms.Bbox.from_bounds(6, 0, 2, 4)) #, clip_on=True)

    ax.patch.set_facecolor('white')
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

    plt.xticks(range(0, 21070, 5000), [str(x) for x in range(0, 21070, 5000)])

    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    tableau20 = [(x[0]/255.0, x[1]/255.0, x[2]/255.0) for x in tableau20]

    plt.xlim(0, 21700)
    plt.ylim(0.67,1.41)
    text_offset = 0.025

    for y in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        plt.plot(range(0, 21700), [y] * len(range(0, 21700)), "--", lw=0.5, color="black", alpha=0.3)

    x_data, y_data = get_scalar('loss/discriminator/total', train_events)
    plt.plot(x_data, y_data, linewidth=1.0, color=tableau20[0])
    plt.text(x_data[-1]+100, y_data[-1], r'{:.3f} Discriminator, Training'.format(y_data[-1]), color=tableau20[0])

    x_data, y_data = get_scalar('loss/discriminator/total', validate_events)
    plt.plot(x_data, y_data, linewidth=1.0, color=tableau20[1])
    plt.text(x_data[-1]+100, y_data[-1]-text_offset, r'{:.3f} Discriminator, Validation'.format(y_data[-1]), color=tableau20[1])

    x_data, y_data = get_scalar('loss/generator/total', train_events)
    plt.plot(x_data, y_data, linewidth=1.0, color=tableau20[2])
    plt.text(x_data[-1]+100, y_data[-1], r'{:.3f} Generator, Training'.format(y_data[-1]), color=tableau20[2])

    x_data, y_data = get_scalar('loss/generator/total', validate_events)
    plt.plot(x_data, y_data, linewidth=1.0, color=tableau20[3])
    plt.text(x_data[-1]+100, y_data[-1]-text_offset, r'{:.3f} Generator, Validation'.format(y_data[-1]), color=tableau20[3])


    # plt.legend([r'Discriminator (Train)', r'Discriminator (Validation)', r'Generator (Train)', r'Generator (Validation)'])
    plt.xlabel(r'\large{Step}')
    plt.ylabel(r'\large{Loss}')
    plt.title(r'\large{\textbf{Generator and Discriminator Loss}}')
    plt.grid(True, linestyle='dotted')
    plt.tight_layout(pad=1)

    # plot_margin = 0.25
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0, x1 + 7000, y0, y1))
    plt.savefig('test.pdf')
    plt.show()

    # # save image to disk
    # vals = hem.get_tag_values('image', 'sampler/images/image/0', train_events)
    # step, clock, image = vals[0] # most recent image
    # print('Found image:', image.height, image.width, image.colorspace)
    # nparr = np.fromstring(image.encoded_image_string, np.uint8)
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    # # img_np = np.reshape(img_np, (256, 2048*8, 3))
    # cv2.imwrite('test.png', img_np)








