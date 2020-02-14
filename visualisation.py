import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D


def visualise_room(graph, dimensions_mean=0, dimensions_std=1):
    """
    Display a graph in the Catalyst visualiser. Displays both the node and
    edge diagram plus a wireframe version of the room.

    Parameters
    ----------

    Returns
    -------
    ax: matplotlib axes
        The axes containing the figure
    """
    fig = plt.figure(0, dpi=90)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = plot_room_graph(ax, graph['nodes'], graph['senders'], graph['receivers'])
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_aspect("equal")

    # Plot frames of each of the objects in the graph
    nodes = np.array(graph['nodes'])
    type_indices = np.argmax(nodes[:, :9], axis=1)
    ax = plot_room(ax, type_indices, nodes[:, 9:11], nodes[:, 11:14],
                   nodes[:, 14:] + dimensions_mean/dimensions_std)
    return ax


def plot_room(ax, type_index, rotations, positions, dimensions):
    """
    Plot a wireframe version of the graph (room) in the axes provided. Each
    node in graph is plotted as a cuboid.

    Parameters
    ----------
    ax: matplotlib axes
        The axes in which to place the figure


    Returns
    -------
    ax: matplotlib axes
        The axes containing the figure
    """
    for type, rot, pos, dim in zip(type_index, rotations, positions, dimensions):
        colour = get_colour(type)
        coords, height = calc_frame_coords(rot, pos, dim)
        plot_frame(ax, coords, pos[2],
                   pos[2] + height, colour)
    axis_equal_3d(ax)
    ax.view_init(elev=90, azim=-90)
    plt.show()


def plot_frame(ax, vertices, base, height, color="b"):
    """
    This function plots the frame of an object, assuming that the object is a 3D
    box.

    Parameters
    ----------
    ax: pyplot object
        pyplot object to plot on.
    vertices: list of 2D tuples
        Ground vertices of the object.
    base: float
        Z coordinate of the base of the frame.
    height: float
        Height of the frame.
    color: python colour
        The colour of the frame.
    """
    for i in range(0, len(vertices) - 1):
        v1 = vertices[i][:2]
        v2 = vertices[i + 1][:2]
        ax.plot3D(*zip(np.array([*v1, base]), np.array([*v2, base])), color=color)
        ax.plot3D(*zip(np.array([*v1, height]), np.array([*v2, height])), color=color)
        ax.plot3D(*zip(np.array([*v1, base]), np.array([*v1, height])), color=color)
        if i == len(vertices) - 2:
            ax.plot3D(*zip(np.array([*v2, base]), np.array([*v2, height])), color=color)


def axis_equal_3d(ax, z_limit=0):
    """
    Equalizes a given axis in 3D, since Matplotlib does not handle it well

    Parameters
    ----------
    ax: pyplot object
        axis to equalize.
    z_limit: float, optional
        we usually want the z axis to have lower limit equal to 0.

    Returns
    -------
    None
    """
    extents = np.array([
        getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']
    )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xy'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    getattr(ax, 'set_zlim')(z_limit, maxsize)


def get_colour(type):
    """
    Use the type to get the colour of an object in a figure

    Parameters
    ----------
    type: integer
        The type of the object: 1 - door, 2 - window, 3 - shower, 4 - bath,
        5 - toilet, 6 - basin, 7 - shower-bath

    Returns
    -------
    colour: hex
        The colour to be used in the figure
    """
    colour = 'b'
    if type == 1:
        colour = '#ff0000'
    if type == 2 or type == 3:
        colour = '#ffc300'
    elif type == 4:
        colour = '#0099ff'
    elif type == 5:
        colour = '#9600ff'
    elif type == 6:
        colour = '#008000'
    elif type == 7:
        colour = '#ff00ff'
    elif type == 8:
        colour = '#33a8ff'
    elif type == 9:
        colour = '#33ffff'
    return colour


def calc_frame_coords(rotation, position, dimensions):
    """
    Calculate the coordinates that define the base of the bounding box of
    a furniture item.

    Parameters
    ----------
    pose: 4 element list
        The pose of the furniture. [1] = x, [2] = y, [3] = z, [4] = angle
    depth: float
        The depth of the bounding box
    width: float
        The width of the bounding box

    Returns
    -------
    coords: list of list
        The coordinates that define the base of the bounding box. Each
        element is of the form [x, z]
    """
    half_depth = dimensions[0]/2
    half_width = dimensions[1]/2
    coords = [[half_depth, half_width, 1],
              [half_depth, -half_width, 1],
              [-half_depth, -half_width, 1],
              [-half_depth, half_width, 1],
              [half_depth, half_width, 1]]
    coords = np.array(coords)
    rot = [rotation, [-rotation[1], rotation[0]]]
    rot = np.concatenate([rot, np.zeros([1, 2])], axis=0)
    pos = np.concatenate([position[:2], np.ones([1])], axis=0)
    pos = pos[:, np.newaxis]
    mat = np.concatenate([rot, pos], axis=1)
    coords = np.dot(mat, coords.T)
    coords = coords[:2].T
    return coords, dimensions[2]


def plot_room_graph(ax, nodes, senders, receivers):
    """
    Plot the graph in the provided axes. Nodes are coloured by their type and
    placed in their (x, z) positions.

    Parameters
    ----------
    ax: matplotlib axes
        The axes in which to place the figure
    node_features: list of lists
        The features that describe the node
    edges: list of lists
        The adjacency matrix for the graph

    Returns
    -------
    ax: matplotlib axes
        The axes containing the figure
    """
    for s, r in zip(senders, receivers):
        ax.plot([nodes[s][-6], nodes[r][-6]],
                [nodes[s][-5], nodes[r][-5]],
                [nodes[s][-4], nodes[r][-4]], c='k')
    for node in nodes:
        colour = get_colour(np.argmax(node[:10]))
        ax.scatter(node[-6], node[-5],  node[-4], c=colour, s=75, edgecolors='k')
    ax.view_init(elev=90, azim=-90)
    return ax
