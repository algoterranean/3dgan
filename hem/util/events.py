import tensorflow as tf
import numpy as np
import cv2
import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib import rc
import matplotlib.pylab as plt
import matplotlib.ticker as tkr
import matplotlib.image as pltimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# set default graph font to LaTeX style
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def get_scalar_tags(events_file):
    tags = set()
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.simple_value:
                    tags.add(v.tag)
    return list(tags)


def get_histogram_tags(events_file):
    tags = set()
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.histo.num:
                    tags.add(v.tag)
    return list(tags)


def get_image_tags(events_file):
    tags = set()
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.image.height:
                    tags.add(v.tag)
    return list(tags)


def get_all_tags(events_file):
    tags = {'images': set(),
            'histograms': set(),
            'scalars': set()}
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.image.height:
                    tags['images'].add(v.tag)
                elif v.histo.num:
                # elif v.histo.num:
                    tags['histograms'].add(v.tag)
                elif v.simple_value is not None:
                # elif v.simple_value:
                    tags['scalars'].add(v.tag)
    return tags


def get_image_values(events_file, tag):
    values = []
    steps = []
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.image.height and v.tag == tag:
                    steps.append(e.step)
                    values.append({'height': v.image.height,
                                   'width': v.image.width,
                                   'colorspace': v.image.colorspace,
                                   'encoded_image': v.image.encoded_image_string})
    return steps, values



def get_scalar_values(events_file, tag):
    values = []
    steps = []
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.simple_value is not None and v.tag == tag:
                    steps.append(e.step)
                    values.append(v.simple_value)
    return steps, values


def get_histogram_values(events_file, tag):
    values = []
    steps = []
    for e in tf.train.summary_iterator(events_file):
        if e.summary:
            for v in e.summary.value:
                if v.histo is not None and v.tag == tag:
                # if v.histo.num and v.tag == tag:
                    steps.append(e.step)
                    values.append({'min': v.histo.min,
                                   'max': v.histo.max,
                                   'num': v.histo.num,
                                   'bucket': list(v.histo.bucket),
                                   'bucket_limit': list(v.histo.bucket_limit)})
    return steps, values


def canvas_to_image(fig, canvas):
    img_w, img_h = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(img_h), int(img_w), 3)
    return image




def get_scalar_plot(events_file, tag, title=None):
    # get scalar data
    x, y = get_scalar_values(events_file, tag)
    # init plot
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # plot literal values
    ax.plot(x, y, alpha=0.2)
    # plot simple moving average
    factor = 10
    smoothed = np.convolve(y, np.ones(factor)/factor, mode='valid')
    max_len = max(len())
    ax.plot(x, smoothed[0:len(x)])

    # set styles
    # title
    title = tag.replace('_', '\_') if title is None else title
    ax.set_title(title, size=14)
    # axis labels
    ax.set_xlabel(r"steps", size=14)
    ax.set_ylabel(r"values", size=14)
    # axis number formatting
    x_format = tkr.FuncFormatter(lambda x, y: '{}k'.format(int(x / 1000)))
    ax.xaxis.set_major_formatter(x_format)
    ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.2e'))
    # grid
    ax.grid(color=(0.9, 0.9, 0.9), linestyle='-')
    # margins
    plt.tight_layout()

    # write to buffer and return
    canvas.draw()
    image = canvas_to_image(fig, canvas)
    plt.close()
    return image



def get_histogram_plot(events_file, tag, title=None):
    # TODO improve this so that it has smoothing
    # TODO improve this so that it has histograms in 3D (entire time series)
    # TODO verify these histograms match tensorboard

    v = get_histogram_values(events_file, tag)
    # last summary
    h = v[1][-1]

    min_val = min(h['bucket_limit'])
    max_val = max(h['bucket_limit'][0:-1])

    # init plot
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    n, bins, _ = ax.hist(x=h['bucket_limit'], weights=h['bucket'], range=(min_val, max_val), bins=int(len(h['bucket'])))

    # from scipy.interpolate import spline
    # xnew = np.linspace(min_val, max_val, 100)
    # power_smooth = spline(h['bucket_limit'][0:-1], h['bucket'][0:-1], xnew)
    # ax.plot(xnew, power_smooth)
    # without smoothing
    plt.plot(h['bucket_limit'][0:-1], h['bucket'][0:-1], alpha=0.75)

    # set styles
    # title
    title = tag.replace('_', '\_') if title is None else title
    ax.set_title(title, size=14)
    # axis labels
    ax.set_xlabel(r"values", size=14)
    ax.set_ylabel(r"distribution", size=14)
    x = h['bucket_limit'][0:-1]
    y = h['bucket'][0:-1]
    # increase frequency of ticks on x axis (use .1 = 10 total)
    x_increment = .1 * (max_val - min_val)
    plt.xticks(np.arange(min_val, max_val + x_increment, x_increment))
    # grid
    ax.grid(color=(0.9, 0.9, 0.9), linestyle='-')
    # margins
    plt.tight_layout()

    # write to buffer and return
    canvas.draw()
    image = canvas_to_image(fig, canvas)
    plt.close()
    return image



# TODO add image summaries
def get_image_plot(events_file, tag, title=None):
    v = get_image_values(events_file, tag)
    i = v[1][-1]
    print(i['width'], i['height'], i['colorspace'])
    with tf.Session() as sess:
        # TODO any way to decode this without using a session?
        i_node = tf.image.decode_image(i['encoded_image'])
        img = i_node.eval()
    return img
    # print(img.shape)

    # i = tf.image.decode_image(v.image.encoded_image_string)
    # img = i.eval()
    # print(v)


if __name__ == '__main__':
    events_file = '/mnt/research/projects/autoencoders/workspace/baseline/iwgan/events.out.tfevents.1498363830.sietch'

    all = get_all_tags(events_file)
    # images = get_image_tags(events_file)
    # histograms = get_histogram_tags(events_file)
    # scalars = get_scalar_tags(events_file)

    print('scalars:', all['scalars'])
    print('images:', all['images'])
    print('histograms:', all['histograms'])

    # for s in all['scalars']:
    #     fn = 'scalar_' + s.replace('/', '.') + '.png'
    #     image = get_scalar_plot(events_file, s)
    #     pltimg.imsave(fn, image)


    # for h in all['histograms']:
    #     fn = 'histogram_' + h.replace('/', '.') + '.png'
    #     image = get_histogram_plot(events_file, h)
    #     pltimg.imsave(fn, image)

    image = get_image_plot(events_file, 'activations/discriminator/c1/lelu/Maximum/montage/image/0')
    print(image.shape)
    image = np.squeeze(image)
    print(image.shape)
    cv2.imwrite('test.png', image)
    # pltimg.imsave('test.png', image)
    # pltimg.imshow(image)

    # for i in all['images']:
    #     fn = 'image_' + h.replace('/', '.') + '.png'
    #     image = get_image_plot(events_file, i)
    #     pltimg.imsave(fn, image)







# simple_values = {}
#
# with tf.Session().as_default():
#     for e in tf.train.summary_iterator(events_file):
#         # print('Event:', e.step, e.wall_time)
#         if e.summary:
#             for v in e.summary.value:
#                 if v.tag not in simple_values:
#                     simple_values[v.tag] = [[], []]
#                 simple_values[v.tag][0].append(e.step)
#                 simple_values[v.tag][1].append(v.simple_value)
#                 # simple_values[v.tag].append((e.step, v.simple_value))
#                 # if v.simple_value:
#                 #     print(v.tag, v.simple_value)
#                 # if v.histo:
#                 #     print(v.tag, v.histo)
#                 if v.image.height:
#                     # i = tf.image.decode_image(v.image.encoded_image_string)
#                     # img = i.eval()
#                     print(e.step, v.tag, v.image.height, v.image.width, v.image.colorspace) #, img.shape, img.dtype)
#
#                 if v.histo:
#                     print(v.tag, v.histo.min, v.histo.max, v.histo.num, v.histo.sum, v.histo.sum_squares)
#                     # the range for a bucket is:
#                     # i == 0: -DBL_MAX .. bucket_limit(0)
#                     # i != 0: bucket_limit(i-1) .. bucket_limit(i)
#                     for b in range(len(v.histo.bucket_limit)):
#                         print('\t', v.histo.bucket_limit[b], v.histo.bucket[b])
#
#                     # print(v.num, v.sum, v.sum_squares)
#
#             # print(v.tag)
#             # if v.simple_value:
#             #     print('\tvalue:', v.simple_value)
#             # elif v.image:
#             #     print('\timage:', v.image)
#             # elif v.histo:
#             #     print('\thisto:', v.histo)
#             # elif v.tensor:
#             #     print('\ttensor:', v.tensor)
#             # # print(v.tag, v.simple_value)
#
# # print(tags)
# print(simple_values['model/loss_g'])
# # plt.plot(simple_values['model/loss_g'][0], simple_values['model/loss_g'][1])
# # plt.show()
#
#
#     # if e.file_version:
#     #     print('file version:', e.file_version)
#     # elif e.graph_def:
#     #     print('graph def:', e.graph_def)
#     # elif e.summary:
#     #     for v in e.summary.value:
#     #         print(v.tag)
#     #     # print('summary:', e.summary)
#     # elif e.log_message:
#     #     print('log message:', e.log_message)
#     # elif e.session_log:
#     #     print('session log:', e.session_log)
#     # elif e.tagged_run_metadata:
#     #     print('tagged run metadata:', e.tagged_run_metadata)
#     # elif e.meta_graph_def:
#     #     print('meta graph def', e.meta_graph_def)
#
#
#
