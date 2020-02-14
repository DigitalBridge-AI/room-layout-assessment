from modules import GNNModel, MLPModel
from create_graph_dataset import projects_to_graph_files
from visualisation import visualise_room
import argparse
import tensorflow as tf
from graph_nets import utils_tf, utils_np
from tqdm import tqdm
import random
import numpy as np
import json
import multiprocessing as mp
import matplotlib.pyplot as plt
import os


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
                    graph['nodes'] = rotate_project(graph['nodes'], random.random() * 2 * np.pi, reflect)
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

def build_parser():
    parser = argparse.ArgumentParser(description="Perform anomaly detection experiments")

    parser.add_argument(
        "-t", "--type",
        help=("The type of model to use, either GNN or MLP"),
        choices=['GNN','MLP'],
        dest="type",
        default='GNN'
    )

    parser.add_argument(
        "-d", "--data_directory",
        help=("The data directory to experiment on. Should contain a "
              "normalisation.json, class_labels.csv and a set of JSON files "
              "describing each of the graphs. Directory is created if it does "
              "if it does not existing and can be created manually with "
              "create_graph_dataset.projects_to_graph_files"),
        dest="data_directory",
        default='Graph_Projects'
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
        default=0.0
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

    return parser


def run_experiment(data_directory, augment, type='GNN', graph_depth=2,
                   outlier_proportion=0.05, mlp_depth=1, mlp_width=16,
                   anomaly_score_size=16):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.isdir('./{}'.format(data_directory)):
        projects_to_graph_files(
            ['./layout_data/train_projects.json'],
            data_directory)

    with open('./{}/normalisation.json'.format(data_directory), 'r') as normalisation_file:
        normalisation = json.load(normalisation_file)
    class_graphs = []
    with open('./{}/class_labels.csv'.format(data_directory), 'r') as class_file:
        line = class_file.readline()
        parts = line.split()
        for i in range(int(parts[1])):
            class_graphs.append([])
        for line in tqdm(class_file):
            parts = line.split()
            class_graphs[int(parts[1])].append(parts[0])

    positive_class = 0
    seed = 0
    random.seed(seed)
    random.shuffle(class_graphs[positive_class])
    train_limit = int(len(class_graphs[0]) * 0.9)
    train_data = class_graphs[positive_class][:train_limit]
    val_positive = class_graphs[positive_class][train_limit:]
    num_training_epochs = 100000
    batch_size = 1024

    tf.reset_default_graph()
    np.random.seed(0)
    tf.set_random_seed(0)

    with open(train_data[0], 'r') as placeholder_file:
        placeholder_data = [json.load(placeholder_file)]
    input_ph = utils_tf.placeholders_from_data_dicts(placeholder_data)

    is_training = tf.placeholder(tf.bool, name='is_training')
    if type == 'GNN':
        model = GNNModel(num_layers=graph_depth, mlp_depth=mlp_depth, mlp_width=mlp_width, anomaly_score_size=anomaly_score_size, is_training=is_training)
    else:
        model = MLPModel(num_layers=graph_depth, mlp_depth=mlp_depth, mlp_width=mlp_width, anomaly_score_size=anomaly_score_size, is_training=is_training)
    room_layout_scores, output = model(input_ph)


    # Training loss.

    with tf.variable_scope('Loss_Variables'):
        radius_squared = tf.Variable(0.5, trainable=False, name='radius_squared')
    loss, data_loss, reg_loss = DSVDD_loss(
        room_layout_scores, radius_squared, outlier_proportion=outlier_proportion)

    # Optimizer.
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.0001,
                                    global_step, 15000,
                                    0.1, staircase=True)

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
        graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
        feed_dict = {input_ph: graphs_tuple, is_training: True}
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
    min_quantile = np.inf
    epoch_last_valid = -1

    model_name = '{}_sb:{}_feature:{}_graphDepth:{}_mlpDepth:{}_mlpWidth:{}'.format(
        type, outlier_proportion, anomaly_score_size, graph_depth, mlp_depth, mlp_width)
    if not augment:
        model_name = model_name + '_na'

    writer = tf.summary.FileWriter('./tensorboard/{}'.format(model_name))
    writer.add_graph(sess.graph)

    iteration = 0
    min_val_loss = np.inf
    for epoch in range(num_training_epochs):
        loss_summ = tf.Summary()
        mean_data_loss = 0
        for batch_counter, batch_graphs in enumerate(data_generator(train_data, batch_size, train=True, augment=augment)):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: True}
            train_values = sess.run({
                    "step": step_op},
                feed_dict=feed_dict)
            iteration += 1
        train_anomaly_scores = []
        mean_data_loss = 0
        for batch_graphs in data_generator(train_data, batch_size):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: False}
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
        # Calculate the loss across the validation set
        positive_anomaly_scores = []
        mean_data_loss = 0
        for batch_graphs in data_generator(val_positive, batch_size):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple, is_training: False}
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
        if iteration > 22500:
            break
        if outlier_proportion > 0 and epoch % 5 == 4:
            value = np.quantile(train_anomaly_scores, 1 - outlier_proportion)
            print("Radius Squared: {}".format(value))
            sess.run(radius_squared.assign(value))


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args.data_directory, args.augment, args.type,
                   int(args.graph_depth), float(args.outlier_proportion),
                   int(args.mlp_depth), int(args.mlp_width),
                   int(args.anomaly_score_size))
