import matplotlib.pyplot as plt
from Settings import Settings
import matplotlib.patches as patches


class Image:
    """
    Create a image class for one data point
    """
    def __init__(self, data, index):
        self.image = data.images[index]
        self.index = index
        self.objects = data.annotations[index]
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.ax.imshow(self.image)

    def draw_one_box(self, boundary):
        """
        add one box to an image
        """
        x_min, y_min, x_max, y_max = boundary["xmin"], boundary["ymin"], boundary["xmax"], boundary["ymax"]
        self.ax.add_patch(
            patches.Rectangle(
                xy=(x_min, y_min),  # point of origin.
                width=x_max - x_min,
                height=y_max - y_min,
                linewidth=2,
                color='red',
                fill=False
            )
        )
        return

    def add_text(self, text, pos):
        """
        add text to certain pos
        :param text:
        :param pos: should be a tuple
        :return:
        """
        self.ax.annotate(text, pos, color='w', weight='bold',
                         fontsize=12, ha='center', va='center')

    @staticmethod
    def show():
        plt.show()

    def save_fig(self, save_name):
        plt.title(save_name)
        plt.savefig("{}Figure{}-{}.png".format(Settings.path_figure, self.index, save_name))
