import numpy as np
import tensorflow as tf
import sonnet as snt
from graph_nets import blocks
from graph_nets import modules


class MLP(snt.AbstractModule):
    def __init__(self, output_size=16, num_layers=1, activate_final=True, is_training=True, name="MLP"):
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

    def _build(self, input_op):
        latent = input_op
        for counter, linear in enumerate(self.linear):
            z = linear(latent)
            if counter != (self.num_layers - 1) or self.activate_final:
                norm = self.bn[counter](z, is_training=self.is_training,
                                        test_local_stats=False)
                latent = tf.nn.leaky_relu(norm, 0.1)
            else:
                latent = z
        return latent


class GNNModel(snt.AbstractModule):
    def __init__(self, num_layers=4, mlp_depth=1, mlp_width=16, anomaly_score_size=16, is_training=True, name="GNNModel"):
        super(GNNModel, self).__init__(name=name)

        self.layers = []
        with tf.variable_scope('Graph_Model'):
            for l in range(num_layers):
                with tf.variable_scope('Layer_{}_Module'.format(l)):
                    self.layers.append(modules.GraphNetwork(
                        edge_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training),
                        node_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training),
                        global_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training)))
        self.readout = MLP(output_size=anomaly_score_size, num_layers=2, activate_final=False, is_training=is_training)
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


class MLPModel(snt.AbstractModule):
    def __init__(self, num_layers=4, mlp_depth=1, mlp_width=16, anomaly_score_size=16, is_training=True, name="MLPModel"):
        super(MLPModel, self).__init__(name=name)

        self.layers = []
        with tf.variable_scope('Graph_Model'):
            # First layer broadcasts the global feature to each of the nodes.
            with tf.variable_scope('Layer_0_Module'):
                self.layers.append(NodeNetwork(
                    node_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training),
                    node_block_opt={'use_received_edges': False, 'use_sent_edges': False, 'use_nodes': True, 'use_globals': True}))
            for l in range(1, num_layers):
                with tf.variable_scope('Layer_{}_Module'.format(l)):
                    self.layers.append(NodeNetwork(
                        node_model_fn=lambda: MLP(output_size=mlp_width, num_layers=mlp_depth, is_training=is_training)))
            self.reducer = blocks.NodesToGlobalsAggregator(tf.unsorted_segment_sum)
        self.readout = MLP(output_size=anomaly_score_size, num_layers=2, activate_final=False, is_training=is_training)
        self.centre = tf.Variable(np.ones(anomaly_score_size, dtype=np.float32), trainable=False, name='centre')

    def _build(self, input_op):
        latent = input_op
        for counter, layer in enumerate(self.layers):
            latent = layer(latent)
            reduced_nodes = self.reducer(latent)
            if counter == 0:
                globals = reduced_nodes
            else:
                globals = tf.concat([globals, reduced_nodes], axis=1)
        output = self.readout(globals)
        diffs = output - self.centre
        squared_diffs = tf.pow(diffs, 2)
        sum_squared_diffs = tf.reduce_sum(squared_diffs, axis=1)
        return sum_squared_diffs, output


class NodeNetwork(snt.AbstractModule):

  def __init__(self,
               node_model_fn,
               node_block_opt={'use_received_edges': False,
                               'use_sent_edges': False,
                               'use_nodes': True,
                               'use_globals': False},
               name="node_network"):

    super(NodeNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn, **node_block_opt)

  def _build(self, graph):
    return self._node_block(graph)
