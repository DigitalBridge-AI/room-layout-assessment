from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import math
import json
import time
import random
import create_dataset
import os
from tqdm import tqdm
import multiprocessing as mp

random.seed(0)


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
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('TkAgg')
    from mpl_toolkits.mplot3d import Axes3D
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
    from catalyst.visualiser.plot_functions import plot_coords, plot_frame, \
        axis_equal_3d, plot_faces
    from matplotlib import pyplot as plt
    for type, rot, pos, dim in zip(type_index, rotations, positions, dimensions):
        colour = get_colour(type)
        coords, height = calc_frame_coords(rot, pos, dim)
        plot_frame(ax, coords, pos[2],
                   pos[2] + height, colour)
        # if type == 1:
        #     cos_angle = np.cos(node[4])
        #     sin_angle = np.sin(node[4])
        #     coords[2][0] +=  cos_angle * node[5]
        #     coords[2][1] -=  sin_angle * node[5]
        #     coords[3][0] +=  cos_angle * node[5]
        #     coords[3][1] -=  sin_angle * node[5]
        #     plot_faces(ax, coords, node[2], 0.1, alpha=0.5, color='grey')

    axis_equal_3d(ax)
    ax.view_init(elev=90, azim=-90)
    plt.show()


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


class MLP(snt.AbstractModule):
    def __init__(self, output_size=16, num_layers=1, activate_final=True, is_training=True, dropout_rate=0.0, name="MLP"):
        super(MLP, self).__init__(name=name)
        self.num_layers = num_layers
        self.linear = []
        self.bn = []
        if isinstance(output_size, int):
            output_size = [output_size] * num_layers
        for n in range(num_layers):
            self.linear.append(snt.Linear(
                output_size=output_size[n], use_bias=False))
            self.bn.append(snt.BatchNorm(offset=False, update_ops_collection=None))
        self.activate_final = activate_final
        self.is_training = is_training
        self.dropout_rate = dropout_rate

    def _build(self, input_op):
        latent = input_op
        for counter, linear in enumerate(self.linear):
            z = linear(latent)
            # tf.summary.histogram("z", z)
            if counter != (self.num_layers - 1) or self.activate_final:
                norm = self.bn[counter](z, is_training=self.is_training,
                                        test_local_stats=False)
                # tf.summary.histogram("norm", norm)
                dropout_norm = tf.nn.dropout(norm, rate=self.dropout_rate)
                latent = tf.nn.leaky_relu(dropout_norm, 0.1)
                # tf.summary.histogram("act", latent)
            else:
                latent = z
        return latent


class GNNModel(snt.AbstractModule):
    def __init__(self, num_layers=4, mlp_depth=1, mlp_width=16, anomaly_score_size=16, is_training=True, dropout_rate=0.0, name="GNNModel"):
        super(GNNModel, self).__init__(name=name)

        self.layers = []
        with tf.variable_scope('Graph_Model'):
            for l in range(num_layers):
                with tf.variable_scope('Layer_{}_Module'.format(l)):
                    self.layers.append(modules.GraphNetwork(
                        edge_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training, dropout_rate=dropout_rate),
                        node_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training, dropout_rate=dropout_rate),
                        global_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training, dropout_rate=dropout_rate)))
        self.readout = MLP(output_size=anomaly_score_size, num_layers=2, activate_final=False, is_training=is_training, dropout_rate=dropout_rate)
        self.centre = tf.Variable(np.ones(anomaly_score_size, dtype=np.float32), trainable=False, name='centre')

    def _build(self, input_op):
        latent = input_op
        for counter, layer in enumerate(self.layers):
            latent = layer(latent)
            if counter == 0:
                globals = latent.globals
            else:
                globals = tf.concat([globals, latent.globals], axis=1)
        output = self.readout(globals)
        diffs = output - self.centre
        squared_diffs = tf.pow(diffs, 2)
        sum_squared_diffs = tf.reduce_sum(squared_diffs, axis=1)
        return sum_squared_diffs, output


def DSVDD_loss(sum_squared_diffs, radius_squared, outlier_proportion=0.1,
               regularisation_weight=1e-5):
    with tf.variable_scope('DSVDD_loss'):
        if outlier_proportion > 0:
            delta = sum_squared_diffs - radius_squared
            max_zero = tf.maximum(0.0, delta)
            data_loss = 1/outlier_proportion * tf.reduce_mean(max_zero)
            loss = radius_squared + data_loss
        else:
            data_loss = tf.reduce_mean(sum_squared_diffs)
            loss = data_loss
        reg_loss = 0
        for t in tf.trainable_variables():
            reg_loss += tf.reduce_sum(tf.pow(t, 2))
        reg_loss = regularisation_weight * reg_loss
        loss = loss + reg_loss
    return loss, data_loss, reg_loss


def data_generator(graphs, batch_size, train=False, augment=False):
    if train:
        random.shuffle(graphs)

    number_of_cores = 4

    input_q = mp.Queue(maxsize=len(graphs)+number_of_cores)
    load_q = mp.Queue(maxsize=batch_size*2)
    output_q = mp.Queue(maxsize=len(graphs)//batch_size + 1)

    def load_graph(input_q, load_q, augment):
        while True:
            input = input_q.get()
            if input is None:
                break
            with open(input, 'r') as graph_file:
                graph = json.load(graph_file)
                if augment:
                    reflect = False
                    if random.random() > 0.5:
                        reflect = True
                    graph['nodes'] = rotate_project(graph['nodes'], random.random() * 2 * math.pi, reflect)
                load_q.put(graph)
        load_q.put(None)

    def queue_to_batch(load_q, output_q, number_of_cores):
        finished_workers = 0
        while True:
            batch = []
            while len(batch) < batch_size:
                graph = load_q.get()
                if graph is None:
                    finished_workers += 1
                    if finished_workers == number_of_cores:
                        if len(batch) > 0:
                            output_q.put(batch)
                        output_q.put(None)
                        break
                else:
                    batch.append(graph)
            output_q.put(batch)

    if train:
        for index in range(len(graphs)//batch_size * batch_size):
            input_q.put(graphs[index])
    else:
        for g in graphs:
            input_q.put(g)
    for _ in range(number_of_cores):
        input_q.put(None)

    pool = mp.Pool(number_of_cores, initializer=load_graph, initargs=(input_q, load_q, augment))
    batch_pool = mp.Pool(1, initializer=queue_to_batch, initargs=(load_q, output_q, number_of_cores))

    while True:
        batch = output_q.get()
        if batch is None:
            break
        yield batch
    pool.close()
    batch_pool.close()


def rotate_project(nodes, angle, reflect):
    nodes = np.array(nodes)
    rot = nodes[:, 9:11]
    pos = nodes[:, 11:13]
    rot = np.concatenate([rot[:, :, np.newaxis], np.concatenate([-rot[:, 1][:, np.newaxis], rot[:, 0][:, np.newaxis]], axis=1)[:, :, np.newaxis]], axis=2)
    rot = np.concatenate([rot, np.zeros([len(nodes), 1, 2])], axis=1)
    pos = np.concatenate([pos, np.ones([len(nodes), 1])], axis=1)
    pos = pos[:, :, np.newaxis]
    mat = np.concatenate([rot, pos], axis=2)
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    rotation_mat = np.array([[cos_angle, -sin_angle, 0],
                             [sin_angle, cos_angle, 0],
                             [0, 0, 1]])
    if reflect:
        rotation_mat[0, 0]  = -rotation_mat[0, 0]
        rotation_mat[1, 1]  = -rotation_mat[1, 1]
    new_mat = np.dot(rotation_mat, mat)
    new_mat = np.transpose(new_mat, [1, 0, 2])
    nodes[:, 9:11] = new_mat[:, 0, :2]
    nodes[:, 11:13] = new_mat[:, :2, 2]
    return nodes


def calculate_rates_at_quantiles(train_anomaly_scores, test_anomaly_scores):
    train = np.array(train_anomaly_scores)
    test = np.array(test_anomaly_scores)
    rates = []
    values = []
    quantiles = []
    for n in range(0, 100):
        quantiles.append(np.quantile(train, n/100))
        rates.append(np.average(test < quantiles[-1]))
        values.append(n/100)
    return rates, values, quantiles


def build_parser():
    parser = argparse.ArgumentParser(description="Perform anomaly detection experiments")

    parser.add_argument(
        "-d", "--dataset",
        help=("The dataset to experiment on"),
        dest="dataset",
        default='Catalyst_Projects'
    )

    parser.add_argument(
        "-c", "--positive_class",
        help="Which class to experiment on",
        dest="positive_class",
        default=0
    )

    parser.add_argument(
        "-a", "--augment",
        help="Whether the room data should NOT be augmented",
        dest="augment",
        action='store_false',
        default=True
    )

    parser.add_argument(
        "-gd", "--graph_depth",
        help="Number of layers in the graph network",
        dest="graph_depth",
        default=2
    )

    parser.add_argument(
        "-o", "--outlier_proportion",
        help="The outlier proportion for Deep SVDD soft boundary",
        dest="outlier_proportion",
        default=0.175
    )

    parser.add_argument(
        "-md", "--mlp_depth",
        help="The depth of the mlps used in the graph network layers",
        dest="mlp_depth",
        default=1
    )

    parser.add_argument(
        "-mw", "--mlp_width",
        help="The width of the mlps used in the graph network layers",
        dest="mlp_width",
        default=16
    )

    parser.add_argument(
        "-as", "--anomaly_score_size",
        help="The number of elements in the final anomaly score",
        dest="anomaly_score_size",
        default=16
    )

    parser.add_argument(
        "-dr", "--dropout_rate",
        help="The dropout rate used during training",
        dest="dropout_rate",
        default=0
    )

    return parser


def run_experiment(dataset_name, positive_class, augment, graph_depth=2,
                   outlier_proportion=0.05, mlp_depth=1, mlp_width=16,
                   anomaly_score_size=16, dropout_rate=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset_name = 'Wilson_Projects_All'
    create_dataset.wilson_projects_to_ind_graph_files(
        ['/home/dhigham/code/catalyst-testing/wilson_data/unique_all_projects.json',
         '/home/dhigham/code/catalyst-testing/wilson_data/simulated_train_projects_positive_overfit.json'],
        dataset_name)

    class_graphs = []
    with open('./{}/class_labels.csv'.format(dataset_name), 'r') as class_file:
        line = class_file.readline()
        parts = line.split()
        for i in range(int(parts[1])):
            class_graphs.append([])
        for line in tqdm(class_file):
            parts = line.split()
            class_graphs[int(parts[1])].append(parts[0])

    negative_class = 1 - positive_class
    seed = 0
    random.seed(seed)
    random.shuffle(class_graphs[positive_class])
    train_limit = int(len(class_graphs[0]) * 0.9)
    train_data = class_graphs[positive_class][:train_limit]
    val_positive = class_graphs[positive_class][train_limit:]
    val_negative = class_graphs[negative_class]
    num_training_epochs = 100000
    batch_size = 1024

    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    with open(train_data[0], 'r') as placeholder_file:
        placeholder_data = [json.load(placeholder_file)]
    input_ph = utils_tf.placeholders_from_data_dicts(placeholder_data)

    is_training = tf.placeholder(tf.bool, name='is_training')
    rate = tf.placeholder(tf.float32, name='rate')
    model = GNNModel(num_layers=graph_depth, mlp_depth=mlp_depth, mlp_width=mlp_width, anomaly_score_size=anomaly_score_size, is_training=is_training, dropout_rate=rate)
    room_layout_scores, output = model(input_ph)


    # Training loss.

    with tf.variable_scope('Loss_Variables'):
        radius_squared = tf.Variable(0.5, trainable=False, name='radius_squared')
    loss, data_loss, reg_loss = DSVDD_loss(
        room_layout_scores, radius_squared, outlier_proportion=outlier_proportion)

    # Optimizer.
    global_step = tf.Variable(0, trainable=False)
    # lr = tf.train.exponential_decay(0.0001,
    #                                 global_step, 15000,
    #                                 0.1, staircase=True)
    lr = 0.001

    lr_summary = tf.summary.scalar("Learning Rate", tensor=lr)
    optimizer = tf.train.AdamOptimizer(lr)

    grads = optimizer.compute_gradients(loss)
    step_op = optimizer.apply_gradients(grads, global_step=global_step)

    saver = tf.train.Saver()

    def make_all_runnable_in_session(*args):
      """Lets an iterable of TF graphs be output from a session as NP graphs."""
      return [utils_tf.make_runnable_in_session(a) for a in args]

    try:
      sess.close()
    except NameError:
      pass
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    init_globals = []
    for batch_graphs in data_generator(train_data, batch_size, augment=augment):
        # visualise_room(batch_graphs[0],  0.8989275984201067, 0.9584436595467952)
        # for b in batch_graphs:
        #     visualise_room(b,  1.20093413606182, 1.0218275245162098)
        #     plt.show()
        graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
        feed_dict = {input_ph: graphs_tuple, is_training: True, rate: 0.0}
        init_values = sess.run({
                "outputs": output
            },
            feed_dict=feed_dict)
        init_globals.extend(init_values['outputs'].tolist())
    centre_value = np.mean(init_globals, axis=0)
    print(centre_value)
    sess.run(model.centre.assign(centre_value))
    if outlier_proportion > 0:
        init_anomaly_scores = np.sum(np.square(init_globals - centre_value), axis=1)
        value = np.quantile(init_anomaly_scores, 1 - outlier_proportion)
        print(value)
        sess.run(radius_squared.assign(value))

    best_epoch = 0
    min_quantile = math.inf
    epoch_last_valid = -1

    model_name = 'sb:{}_feature:{}_graphDepth:{}_mlpDepth:{}_mlpWidth:{}_dropout:{}_unique'.format(
        outlier_proportion, anomaly_score_size, graph_depth, mlp_depth, mlp_width, dropout_rate)
    if not augment:
        model_name = model_name + '_na'

    writer = tf.summary.FileWriter('./tensorboard/{}'.format(model_name))
    writer.add_graph(sess.graph)

    # for counter, t in enumerate(tf.trainable_variables()):
    #     weights = sess.run(t)
    #     if np.ndim(weights) > 1:
    #         max = np.max(np.abs(weights))
    #         plt.imshow(weights)
    #         plt.set_cmap('bwr')
    #         plt.colorbar()
    #         plt.clim(-max, max)
    #         plt.savefig('initial_{}.png'.format(t.name.replace("/", "_")))
    #         plt.close()

    iteration = 0
    min_val_loss = math.inf
    for epoch in range(num_training_epochs):
        loss_summ = tf.Summary()
        mean_data_loss = 0
        for batch_counter, batch_graphs in enumerate(data_generator(train_data, batch_size, train=True, augment=augment)):
            # visualise_room(batch_graphs[0], 1.1241034387734892, 1.0013599372364184)
            # plt.show()
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: True, rate: dropout_rate}
            train_values = sess.run({
                    "step": step_op},
                feed_dict=feed_dict)
            iteration += 1
        train_anomaly_scores = []
        mean_data_loss = 0
        for batch_graphs in data_generator(train_data, batch_size):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: False, rate: 0.0}
            max_train_values = sess.run({"loss": loss,
                                         "data_loss": data_loss,
                                         "anomaly_value": room_layout_scores},
                                         feed_dict=feed_dict)
            train_anomaly_scores.extend(max_train_values["anomaly_value"].tolist())
            mean_data_loss += len(batch_graphs) * max_train_values['data_loss']
        train_data_loss = mean_data_loss/len(train_data)
        train_loss = train_data_loss + (max_train_values['loss'] - max_train_values['data_loss'])
        loss_summ.value.add(tag="Train Loss", simple_value=train_loss)
        loss_summ.value.add(tag="Train Data Loss", simple_value=train_data_loss)
        # Calculate the loss across the positive validation set
        positive_anomaly_scores = []
        mean_data_loss = 0
        for batch_graphs in data_generator(val_positive, batch_size):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: False, rate: 0.0}
            positive_values = sess.run({"loss": loss,
                                        "data_loss": data_loss,
                                        "reg_loss": reg_loss,
                                        "anomaly_value": room_layout_scores},
                                        feed_dict=feed_dict)
            positive_anomaly_scores.extend(positive_values["anomaly_value"].tolist())
            mean_data_loss += len(batch_graphs) * positive_values['data_loss']
        val_data_loss = mean_data_loss / len(val_positive)
        val_loss = val_data_loss + (positive_values['loss'] - positive_values['data_loss'])
        loss_summ.value.add(tag="Num Iterations", simple_value=iteration)
        loss_summ.value.add(tag="Regularisation Loss", simple_value=positive_values['reg_loss'])
        loss_summ.value.add(tag="Val Loss", simple_value=val_loss)
        loss_summ.value.add(tag="Val Data Loss", simple_value=val_data_loss)


        print("{}: Train Loss: {}, Val Loss: {}, Reg Loss: {}".format(
            epoch, train_loss, val_loss, positive_values['reg_loss']))
        writer.add_summary(loss_summ, epoch)
        writer.flush()
        train_anomaly_scores = None
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if not os.path.isdir('./{}/checkpoint'.format(model_name)):
                os.mkdir('./{}'.format(model_name))
                os.mkdir('./{}/checkpoint'.format(model_name))
            save_path = saver.save(sess, './{}/checkpoint/model.ckpt'.format(model_name))
            # Calculate the anomaly scores across the training set
            # train_anomaly_scores = []
            # mean_data_loss = 0
            # for batch_graphs in data_generator(train_data, batch_size):
            #     graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            #     feed_dict = {input_ph: graphs_tuple, is_training: False, rate: 0.0}
            #     max_train_values = sess.run({"anomaly_value": squared_diffs},
            #                                 feed_dict=feed_dict)
            #     train_anomaly_scores.extend(max_train_values["anomaly_value"].tolist())
            # positive_rates, thresholds, _ = calculate_rates_at_quantiles(train_anomaly_scores, positive_anomaly_scores)
            # plt.plot(thresholds, positive_rates)
            # # Calculate the false positive rate across the negative set
            # negative_anomaly_scores = []
            # for batch_graphs in data_generator(val_negative, batch_size):
            #     graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            #     feed_dict = {input_ph: graphs_tuple, is_training: False, rate: 0.0}
            #     negative_values = sess.run({"anomaly_value": squared_diffs},
            #                                feed_dict=feed_dict)
            #     negative_anomaly_scores.extend(negative_values["anomaly_value"].tolist())
            # negative_rates, _, _ = calculate_rates_at_quantiles(train_anomaly_scores, negative_anomaly_scores)
            # plt.plot(thresholds, negative_rates)
            # plt.legend(['Positive Rates', 'Negative Rates'])
            # plt.savefig('./{}/rates.png'.format(model_name))
            # plt.close()
        if iteration > 1000000:
            break
        if outlier_proportion > 0 and epoch % 5 == 4:
            if train_anomaly_scores is None:
                train_anomaly_scores = []
                mean_data_loss = 0
                for batch_graphs in data_generator(train_data, batch_size):
                    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
                    feed_dict = {input_ph: graphs_tuple, is_training: False, rate: 0.0}
                    max_train_values = sess.run({"anomaly_value": room_layout_scores},
                                                feed_dict=feed_dict)
                    train_anomaly_scores.extend(max_train_values["anomaly_value"].tolist())
            value = np.quantile(train_anomaly_scores, 1 - outlier_proportion)
            print("Radius Squared: {}".format(value))
            sess.run(radius_squared.assign(value))


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args.dataset, int(args.positive_class), args.augment,
                   int(args.graph_depth), float(args.outlier_proportion),
                   int(args.mlp_depth), int(args.mlp_width),
                   int(args.anomaly_score_size), float(args.dropout_rate))
