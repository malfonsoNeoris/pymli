import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as clrs

from mpl_toolkits.mplot3d import Axes3D


def scatter3d(x, y, z, title='3D Graph', xlabel='X', ylabel='Y', zlabel='Z', cs=None, cslabel=None):
    cm = plt.get_cmap('jet')
    scalar_map = None
    if cs is not None:
        c_norm = clrs.Normalize(vmin=min(cs), vmax=max(cs))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    if cs is not None:
        ax.scatter(x, y, z, c=scalar_map.to_rgba(cs))
        scalar_map.set_array(cs)
        fig.colorbar(scalar_map)
    else:
        ax.scatter(x, y, z)
    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if cslabel is not None:
        ax.set_label(cslabel)
    plt.show()
