import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class BenchmarkBarPlot:
    def __init__(self, num_classes, num_bars, class_labels=None, bar_labels=None, bar_width=0.2, stretch=1.1):
        ''' A typical plot with `num_bars` for each class:
            for example, a class could be a type of dataset,
            and a bar could be the average performance on
            the dataset for a given algorithm.
            Typically num_bars < num_classes, num_bars < 10
        '''
        if bar_labels is None:
            bar_labels = [i for i in range(num_bars)]
        if class_labels is None:
            class_labels = [i for i in range(num_classes)]

        if not isinstance(bar_labels,list):
            raise ValueError("Waiting for list of labels (strings)",bar_labels)
        if not isinstance(class_labels,list):
            raise ValueError("Waiting for list of labels (strings)",class_labels)


        assert num_bars == len(bar_labels)
        assert num_bars < 10 ; assert num_classes < 20

        self.class_labels = class_labels ; self.bar_labels = bar_labels
        self.num_bars = num_bars ; self.num_classes = num_classes
        self.colors = plt.cm.tab20.colors
        self.fig, self.ax = plt.subplots()
        self.bar_handles = [Line2D([0], [0], color=c, lw=6, label=lab) for c,lab in zip(self.colors,self.bar_labels)]
        #label locations, rescale depending on total number of bars
        self.locs = stretch*np.arange(len(class_labels))
        # the width of the bars
        self.bar_width = bar_width

    def plot_bars(self, data, y_label="", title="", y_error=None):
        ''' Method to display `num_bars` bars for each class.
            `data` is [num_bars,num_classes] shaped array.
            if y errors are included, they should match `data` shape.

            In case the bars are too close or overlapping, one needs to
            tweak the `bar_width` and `stretch` parameters
            (stretch corresponds to the length of the x-axis)
        '''
        # helps positioning each bar inside classes
        assert data.shape[0] == self.num_bars
        assert data.shape[1] == self.num_classes
        if y_error is not None:
            assert y_error.shape == data.shape
            # for each bar type, for each class
            # there should be a y_error if not None

        for i,label in enumerate(self.bar_labels):
            # shift each class by a bar width
            self.ax.bar(self.locs + i*self.bar_width, data[i], width=self.bar_width,
                        label=label, color=self.colors[i], yerr=y_error, align='edge')

        y_low, y_up, y_range = data.min() , data.max(), data.max() - data.min()

        y_ticks = np.arange(1,11)/10*(y_up.round(0)) # round(i) is the # of decimal places
        self.ax.set_ylabel(y_label, fontsize=10)
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xticks(self.locs+self.num_bars*self.bar_width/2)
        self.ax.set_xticklabels(self.class_labels)
        self.ax.yaxis.grid(linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.ax.legend(handles=self.bar_handles,bbox_to_anchor=(1, 1), ncol=2, fancybox=True, fontsize=12,)
        plt.show()
