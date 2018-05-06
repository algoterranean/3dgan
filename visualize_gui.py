import tensorflow as tf
from tensorflow.tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
# event_accumulator, event_file_inspector, event_file_loader, event_multiplexer
import numpy as np
import wx
import os
import re
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.backends.backend_wx import _load_bitmap
from matplotlib.figure import Figure


class ProjectFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title='Tensorflow Project', pos=(50,50), size=(900,500))
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        # create menu
        menu_bar = wx.MenuBar()
        menu = wx.Menu()
        m_open = menu.Append(-1, "&Open\tCtrl+O", "Open project directory")
        self.Bind(wx.EVT_MENU, self.OnOpen, m_open)
        m_exit = menu.Append(-1, "&Quit\tCtrl+Q", "Close window and exit program")
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        menu_bar.Append(menu, "&File")
        self.SetMenuBar(menu_bar)
        # create status bar
        self.CreateStatusBar()
        # create basic layout
        project_panel = wx.Panel(self)
        project_sizer = wx.BoxSizer(wx.VERTICAL)
        project_panel.SetSizer(project_sizer)
        root_sizer = wx.BoxSizer(wx.HORIZONTAL)
        root_sizer.Add(project_panel, 1, wx.EXPAND)
        # create project tree
        self.project_tree = wx.TreeCtrl(project_panel, style=wx.TR_MULTIPLE|wx.TR_HAS_BUTTONS)
        self.project_tree.Bind(wx.EVT_TREE_SEL_CHANGED, self.OnSelectChanged)



        # n = project_tree.AddRoot("Root Node")
        # project_tree.AppendItem(n, "hello")
        project_sizer.Add(self.project_tree, 1, wx.EXPAND)
        # fill out remaining space for now
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)
        root_sizer.Add(self.canvas, 2, wx.EXPAND)

        # axes = self.figure.add_subplot(111)
        # t = np.linspace(0.0, 3.0, 0.01)
        # s = np.sin(2 * np.pi * t)
        # axes.plot(t, s)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        # layout
        self.SetAutoLayout(True)
        self.SetSizer(root_sizer)
        self.Layout()

    def OnSelectChanged(self, event):
        selected = self.project_tree.GetSelections()
        selected_scalars = []

        print('SelectChanged:', selected)
        for s in selected:
            data = self.project_tree.GetItemData(s)
            if 'type' in data:
                if data['type'] == 'scalar':
                    scalars = self.em.Scalars('train', data['tag'])
                    selected_scalars.append((data['tag'], scalars))
        if len(selected_scalars) > 0:
            self.PlotScalars(selected_scalars)

    def OnPaint(self, event):
        self.canvas.draw()
        event.Skip()

    def OnClose(self, event):
        self.Destroy()

    def OnOpen(self, event):
        dialog = wx.DirDialog(self, "Open Project", style=wx.DD_DEFAULT_STYLE)
        if dialog.ShowModal() == wx.ID_CANCEL:
            return
        self.project_path = dialog.GetPath()
        self.LoadProject(self.project_path)

    def PlotScalars(self, scalars):
        self.figure.clear()
        for s in scalars:
            tag = s[0]
            data = s[1]
            axes = self.figure.add_subplot(111)
            xs = [s[2] for s in data]
            ys = [s[1] for s in data]
            axes.plot(ys, xs)
        self.canvas.draw()

    def LoadProject(self, path):
        self.project_tree.DeleteAllItems()
        project_name = path.split('/')[-1]
        root_node = self.project_tree.AddRoot(project_name)

        checkpoint_files = []
        event_files = []
        graph_file = None
        options_file = None

        model_node = None

        for root, dirs, files in os.walk(path):
            new_path = root.split(os.sep)
            # print(os.path.basename(root))

            if 'options.config' in files or 'graph.pbtxt' in files:
                model_node = self.project_tree.AppendItem(root_node, new_path[-1])
                graph_node = self.project_tree.AppendItem(model_node, "Graph")
                options_node = self.project_tree.AppendItem(model_node, "Options")
                events_node = self.project_tree.AppendItem(model_node, "Events")
                checkpoints_node = self.project_tree.AppendItem(model_node, "Checkpoints")

            for file in files:
                r = re.match('(checkpoint-[0-9]+).meta', file)
                if r:
                    checkpoint_files.append(r.group(1))
                    self.project_tree.AppendItem(checkpoints_node, r.group(1))
                elif file == 'graph.pbtxt':
                    graph_file = os.path.join(root, file)
                    self.project_tree.AppendItem(graph_node, file)
                elif file == 'options.config':
                    options_file = os.path.join(root, file)
                    self.project_tree.AppendItem(options_node, file)
                elif re.match('events.out.*', file):
                    event_files.append(os.path.join(root, file))
                    self.project_tree.AppendItem(events_node, file)

        unique_subdirs = set([os.path.dirname(fn) for fn in event_files])
        self.em = EventMultiplexer()
        for d in unique_subdirs:
            self.em.AddRunsFromDirectory(d)







        # graph_node = self.project_tree.AppendItem(root_node, "Graph")
        # options_node = self.project_tree.AppendItem(root_node, "Options")
        # event_node = self.project_tree.AppendItem(root_node, "Events")
        #
        # unique_subdirs = []
        # for x in event_files:
        #     subdir = x.split('/')[-2]
        #     if subdir not in unique_subdirs:
        #         unique_subdirs.append(subdir)
        #         # self.project_tree.AppendItem(event_node, subdir)
        #
        # scalar_node = self.project_tree.AppendItem(event_node, "Scalars")
        # histogram_node = self.project_tree.AppendItem(event_node, "Histograms")
        # image_node = self.project_tree.AppendItem(event_node, "Images")
        #
        # unique_scalars = []
        # unique_histograms = []
        # unique_images = []
        #
        # self.em = EventMultiplexer()
        # self.em.AddRunsFromDirectory(path)
        # self.em.Reload()
        # for k, v in self.em.Runs().items():
        #     print('run:', k)
        #     for e in v['histograms']:
        #         if e not in unique_histograms:
        #             unique_histograms.append(e)
        #     for e in v['scalars']:
        #         if e not in unique_scalars:
        #             unique_scalars.append(e)
        #     for e in v['images']:
        #         if e not in unique_images:
        #             unique_images.append(e)
        #
        # unique_scalars.sort()
        # unique_histograms.sort()
        # unique_images.sort()
        #
        # for s in unique_scalars:
        #     self.project_tree.AppendItem(scalar_node, s, data={'type': 'scalar', 'tag': s})
        # for h in unique_histograms:
        #     self.project_tree.AppendItem(histogram_node, h, data={'type': 'histogram', 'tag': h})
        # for i in unique_images:
        #     self.project_tree.AppendItem(image_node, i, data={'type': 'image', 'tag': i})




        # print('images:', unique_images)
        # images = em.Images('train', 'fake/images/image/0')
        # print('found', len(images), 'images')
        # for i in images:
        #     # wall_time, step, encoded_image_string, width, height
        #     print(i[0], i[1], '({}, {})'.format(i[3], i[4]))
        #
        # print('scalars:', unique_scalars)
        # scalars = em.Scalars('train', 'loss/generator/total')
        # print('found', len(scalars), 'scalars')
        # for s in scalars:
        #     print(s[0], s[1], s[2])




        # images: wall_time, step, encoded_image_string, width, height
        # scalars: wall_time, step, value
        # histogram: wall_time, step, histogram_value
        # histogram_value: min, max, num, sum, sum_squares, bucket_limit, bucket


        # for tag in unique_images:
        #     image_events = em.Images('train', tag)
        # print(image_events)



if __name__ == '__main__':
    a = wx.App()
    f = ProjectFrame()
    f.Show()
    a.MainLoop()
