import os
import argparse
import json
import random
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from graph_nets import utils_tf, utils_np

from modules import GNNModel, MLPModel
from create_graph_dataset import projects_to_graph_files
from visualisation import visualise_room

def data_generator(graphs, batch_size):
    number_of_cores = 1

    input_q = mp.Queue(maxsize=len(graphs)+number_of_cores)
    load_q = mp.Queue(maxsize=batch_size*2)
    output_q = mp.Queue(maxsize=len(graphs)//batch_size + 1)

    def load_graph(input_q, load_q):
        while True:
            input = input_q.get()
            if input is None:
                break
            with open(input, 'r') as graph_file:
                graph = json.load(graph_file)
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

    for g in graphs:
        input_q.put(g)
    for _ in range(number_of_cores):
        input_q.put(None)

    pool = mp.Pool(number_of_cores, initializer=load_graph, initargs=(input_q, load_q))
    batch_pool = mp.Pool(1, initializer=queue_to_batch, initargs=(load_q, output_q, number_of_cores))

    while True:
        batch = output_q.get()
        if batch is None:
            break
        yield batch
    pool.close()
    batch_pool.close()


def calculate_ROC_values(positive_anomaly_scores, negative_anomaly_scores):
    positive = np.array(positive_anomaly_scores)
    negative = np.array(negative_anomaly_scores)
    rates = [0]
    values = [0]
    auc = 0
    for n in range(1, 101):
        quantile = np.quantile(negative, n/100)
        rates.append(np.average(positive < quantile))
        values.append(n/100)
        if n > 1:
            auc += (rates[-2] + rates[-1])/2 * 0.01

    return rates, values, auc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test anomaly detection model over a number of different "
                    "data_directorys")

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
              "create_graph_data_directory.projects_to_graph_files"),
        dest="data_directory",
        default='Graph_Projects_Testing'
    )

    parser.add_argument(
        "-gd", "--graph_depth",
        help="Number of layers in the graph network",
        dest="graph_depth",
        default=2
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
        "-c", "--checkpoint_directory",
        help="The directory which contains the checkpoint files",
        dest="checkpoint",
        required=True
    )

    parser.add_argument(
        "-o", "--output_name",
        help="The file name for the output ROC curve plot",
        dest="output_name",
        required=True
    )

    return parser


def run_experiment(data_directory, checkpoint, output_name, type='GNN',
                   graph_depth=2, mlp_depth=1, mlp_width=16,
                   anomaly_score_size=16):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.isdir('./{}'.format(data_directory)):
        projects_to_graph_files(
            ['./layout_data/train_projects.json',
             './layout_data/valid_projects.json',
             './layout_data/valid_projects.json'],
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

    seed = 0
    random.seed(seed)
    random.shuffle(class_graphs[0])
    train_limit = int(len(class_graphs[0]) * 0.9)
    test_positive = class_graphs[0][train_limit:]
    batch_size = 1024

    with open(test_positive[0], 'r') as placeholder_file:
        placeholder_data = [json.load(placeholder_file)]
    input_ph = utils_tf.placeholders_from_data_dicts(placeholder_data)

    if type == 'GNN':
        model = GNNModel(num_layers=graph_depth, mlp_depth=mlp_depth, mlp_width=mlp_width, anomaly_score_size=anomaly_score_size, is_training=False)
    else:
        model = MLPModel(num_layers=graph_depth, mlp_depth=mlp_depth, mlp_width=mlp_width, anomaly_score_size=anomaly_score_size, is_training=False)
    room_layout_scores, _ = model(input_ph)

    def make_all_runnable_in_session(*args):
      """Lets an iterable of TF graphs be output from a session as NP graphs."""
      return [utils_tf.make_runnable_in_session(a) for a in args]

    try:
      sess.close()
    except NameError:
      pass
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "{}/model.ckpt".format(checkpoint))

    # Calculate the room layout scores for the validation set
    positive_anomaly_scores = []
    num_true_positives = 0
    mean_data_loss = 0
    for batch_graphs in data_generator(test_positive, batch_size):
        graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
        feed_dict = {input_ph: graphs_tuple}
        positive_values = sess.run({"anomaly_value": room_layout_scores},
                                    feed_dict=feed_dict)
        positive_anomaly_scores.extend(positive_values["anomaly_value"].tolist())

    positive_anomaly_scores = np.squeeze(positive_anomaly_scores)
    test_positive = np.array(test_positive)
    indices = np.argsort(positive_anomaly_scores)
    positive_anomaly_scores = positive_anomaly_scores[indices]
    test_positive = test_positive[indices]
    min_score = positive_anomaly_scores[0]
    max_score = positive_anomaly_scores[-1]

    # Calculate the room layout scores for each of the negative sets
    negative_anomaly_scores = [[]]
    labels = ['', 'Valid Rooms', 'Empty Rooms']
    for c in range(1, len(class_graphs)):
        negative_anomaly_scores.append([])
        num_false_positives = 0
        for batch_graphs in data_generator(class_graphs[c], batch_size):
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(batch_graphs)
            feed_dict = {input_ph: graphs_tuple}
            negative_values = sess.run({"anomaly_value": room_layout_scores},
                                        feed_dict=feed_dict)
            negative_anomaly_scores[c].extend(negative_values["anomaly_value"].tolist())
        negative_anomaly_scores[c] = np.squeeze(negative_anomaly_scores[c])
        class_graphs[c] = np.array(class_graphs[c])
        indices = np.argsort(negative_anomaly_scores[c])
        negative_anomaly_scores[c] = negative_anomaly_scores[c][indices]
        class_graphs[c] = class_graphs[c][indices]
        min_score = min(min_score, negative_anomaly_scores[c][0])
        max_score = max(max_score, negative_anomaly_scores[c][-1])
        negative_rates, thresholds, auc = calculate_ROC_values(positive_anomaly_scores, negative_anomaly_scores[c])
        labels[c] = '{} - {:3f}'.format(labels[c], auc)
        plt.plot(thresholds, negative_rates)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(labels[1:])
    if not os.path.isdir('./roc_curves'):
        os.mkdir('./roc_curves')
    plt.savefig('./roc_curves/{}'.format(output_name))
    plt.close()

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args.data_directory, args.checkpoint, args.output_name,
                   args.type, int(args.graph_depth), int(args.mlp_depth),
                   int(args.mlp_width), int(args.anomaly_score_size))
