import os
from tqdm import tqdm
import shutil
import json
import numpy as np

from shapely.geometry import LineString, Point


def projects_to_graph_files(files, dataset_name):
    valid_furniture = ['door', 'window', 'shower', 'bath', 'toilet', 'basin',
                       'shower-bath']
    # The number of types is the len(valid_furniture) + 3 (extrac types: floor,
    # wall, other)
    num_node_types = len(valid_furniture) + 2
    if os.path.isdir('./{}'.format(dataset_name)):
        shutil.rmtree('./{}'.format(dataset_name))
    os.mkdir('./{}'.format(dataset_name))
    project_counter = 0
    class_labels = []
    projects = []
    all_sizes = []
    all_edges = []
    all_globals = []
    for label, f in enumerate(files):
        with open(f, 'r') as lines:
            for line in tqdm(lines):
                layout_dict = json.loads(line)
                include_products = True
                if label == 2:
                    include_products = False
                graph, shift = convert_project(layout_dict, include_products)
                # visualise_room(graph)
                projects.append(graph)
                class_labels.append(label)
                project_counter += 1
                if label == 0:
                    all_globals.append(graph['globals'])
                    for node in graph['nodes']:
                        all_sizes.append(node[-3:])
    all_sizes = np.array(all_sizes)
    # all_edges = np.array(all_edges)
    all_globals = np.array(all_globals)
    all_sizes_mean = np.mean(all_sizes)
    all_sizes_std = np.std(all_sizes)
    print("Sizes Stats: Mean: {}, STD: {}".format(all_sizes_mean, all_sizes_std))
    # all_edges_mean = np.mean(all_edges)
    # all_edges_std = np.std(all_edges)
    all_globals_mean = np.mean(all_globals, axis=0)
    all_globals_std = np.std(all_globals, axis=0)
    print("Globals Stats: Mean: {}, STD: {}".format(all_globals_mean, all_globals_std))
    normalisation = {'sizes_mean': all_sizes_mean.item(), 'sizes_std': all_sizes_std.item(),
                     'globals_mean': all_globals_mean.tolist(),
                     'globals_std': all_globals_std.tolist()}
    with open('./{}/normalisation.json'.format(dataset_name), 'w') as normalisation_file:
        json.dump(normalisation, normalisation_file)
    with open('./{}/class_labels.csv'.format(dataset_name), 'w') as class_file:
        class_file.write('Num_classes: {}\n'.format(len(files)))
        for counter, p in tqdm(enumerate(projects)):
            class_file.write('./{}/{}.json {}\n'.format(
                 dataset_name, counter, class_labels[counter]))
            # visualise_room(p)
            # plt.show()
            nodes = np.array(p['nodes'])
            nodes[:, -4] = nodes[:, -4] - 1.2
            nodes[:, -3:] = nodes[:, -3:] - all_sizes_mean
            nodes[:, -6:] = nodes[:, -6:]/all_sizes_std
            p['nodes'] = nodes.tolist()
            graph_globals = np.array(p['globals'])
            graph_globals = (graph_globals - all_globals_mean)/all_globals_std
            p['globals'] = graph_globals.tolist()
            nodes = np.array(p['nodes'])
            # visualise_room(p,  1.20093413606182, 1.0218275245162098)
            with open('./{}/{}.json'.format(dataset_name, counter), 'w') as graph_file:
                json.dump(p, graph_file)


def convert_project(layout, include_products=True, include_apertures=True):
    valid_furniture = ['door', 'window', 'shower', 'bath', 'toilet', 'basin',
                       'shower-bath']
    # The number of types is the len(valid_furniture) + 3 (extra types: floor,
    # wall, other)
    num_node_types = len(valid_furniture) + 2
    nodes, senders, receivers, graph_globals, shift, line_strings = create_wall_nodes(
        layout['room']['vertices'], layout['room']['height'],
        num_node_types)
    occupied_area = 0
    for p in layout['products']:
        for counter, f in enumerate(valid_furniture):
            if p['type'] == f:
                break
        else:
            continue
            counter += 1
        if not include_apertures and (counter == 0 or counter == 1):
            continue
        if not include_products and counter > 1:
            continue
        node = [0] * num_node_types
        node[counter + 2] = 1
        if not isinstance(p['rotation'], list):
            rotation = convert_angles_to_rotation(p['rotation'])
        else:
            rotation = [p['rotation'][0], p['rotation'][1]]
        node.extend(rotation)
        x = p['position']['x'] - shift[0]
        y = p['position']['y']
        z = p['position']['z'] - shift[1]
        wall_point = np.dot([rotation, [-rotation[1], rotation[0]]], [p['dimension']['depth']/2, 0])
        wall_x = x -  wall_point[0]
        wall_z = z - wall_point[1]
        if abs(p['position']['y']) < 0.0001:
            senders.append(0)
            receivers.append(len(nodes))
            senders.append(len(nodes))
            receivers.append(0)
        wall_point = Point(wall_x, wall_z)
        for wall_counter, l in enumerate(line_strings):
            dist = l.distance(wall_point)
            if abs(dist) < 0.05:
                senders.append(wall_counter + 1)
                receivers.append(len(nodes))
                senders.append(len(nodes))
                receivers.append(wall_counter + 1)
                continue
            side_dist = dist - p['dimension']['width']/2
            if abs(side_dist) < 0.05:
                senders.append(wall_counter + 1)
                receivers.append(len(nodes))
                senders.append(len(nodes))
                receivers.append(wall_counter + 1)
        if (counter == 0 or counter == 1):
            x = wall_x
            z = wall_z
        else:
            if p['position']['y'] == 0:
                occupied_area += (p['dimension']['depth']
                                  * p['dimension']['width'])
        node.extend([x, z, y])
        node.extend([p['dimension']['depth'],
                     p['dimension']['width'],
                     p['dimension']['height']])
        nodes.append(node)
    data_dict = {}
    data_dict['nodes'] = nodes
    edges = []
    for _ in senders:
        edges.append([0.0])
    data_dict['senders'] = senders
    data_dict['receivers'] = receivers
    data_dict['edges'] = edges
    graph_globals.append(occupied_area/graph_globals[0])
    data_dict['globals'] = graph_globals
    return data_dict, shift


def create_wall_nodes(vertices, height, num_node_types):

    room_features = []
    senders = []
    receivers = []
    corner_points = []
    for v in vertices:
        corner_points.append([v['x'], v['z']])
    line_strings = []
    corner_points = np.array(corner_points)
    min_corner_points = np.min(corner_points, axis=0)
    max_corner_points = np.max(corner_points, axis=0)
    shift = (min_corner_points + max_corner_points)/2
    corner_points = corner_points - shift
    corner_points_rolled = np.roll(corner_points, -1, axis=0)
    room_area = np.sum(((corner_points[:, 0] + corner_points_rolled[:, 0]) *
                        (corner_points[:, 1] - corner_points_rolled[:, 1])))/2
    line_strings = []
    for p1, p2 in zip(corner_points, corner_points_rolled):
        line_strings.append(LineString(coordinates=[[p1[0], p1[1]], [p2[0], p2[1]]]))
    line_vectors = corner_points_rolled - corner_points
    wall_angles = np.arctan2(line_vectors[:, 0], line_vectors[:, 1])
    wall_points = corner_points + line_vectors/2
    wall_dimensions = np.linalg.norm(line_vectors, axis=1)

    # Add floor node
    floor_feature = [0] * num_node_types
    floor_feature[0] = 1
    floor_feature.extend([1, 0])
    floor_centre = (min_corner_points + max_corner_points)/2
    floor_feature.extend([0, 0, 0])
    floor_dimensions = max_corner_points - min_corner_points
    floor_feature.extend([floor_dimensions[0], floor_dimensions[1], 0])
    room_features.append(floor_feature)
    # Add wall nodes
    for counter, a in enumerate(wall_angles):
        senders.append(0)
        receivers.append(counter + 1)
        senders.append(counter + 1)
        receivers.append(0)
        previous = counter - 1
        if previous == -1:
            previous = len(corner_points) - 1
        next = counter + 1
        if next == len(corner_points):
            next = 0
        senders.append(counter + 1)
        receivers.append(previous + 1)
        senders.append(counter + 1)
        receivers.append(next + 1)
        wall_feature = [0] * num_node_types
        wall_feature[1] = 1
        wall_feature.extend(convert_angles_to_rotation(a))
        wall_feature.extend([wall_points[counter, 0], wall_points[counter, 1],
                             0])
        wall_feature.extend([0, wall_dimensions[counter], height])
        room_features.append(wall_feature)
    graph_globals = [room_area, np.sum(wall_dimensions), len(wall_dimensions)]
    return room_features, senders, receivers, graph_globals, shift, line_strings


def convert_angles_to_rotation(angle):

    sin = np.sin(angle)
    cos = np.cos(angle)
    rotation = [cos, sin]
    return rotation
