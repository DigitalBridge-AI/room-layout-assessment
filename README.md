# room-layout-assessment
Room layout assessment using the a Graph Neural Network and the Deep Support Vector Data Description loss. Code for the paper 'A One-Class Graph Neural Network for Room Layout Assessment' currently under review for ICML 2020.

Contact jacob.rawling@digitalbridge.com for comments and questions.

## Installation

The package can be installed by:

```shell
$ pip install -e .
```

The code has a dependency on DeepMind's [Graph Nets](https://github.com/deepmind/graph_nets).

The code was developed using python 3.5.2.

## Reconstitute the training Data

Github limits files to 100MB. To get around this we have split the file into 3. Before training the 3 parts must be reconstituted into a parent file by:

```shell
$ cd layout_data
$ python reconstitute_train_file.py
```

## Train a Network

The room layout assessment model can be trained by:

```shell
$ python train_room_layout.py
```

The room layout assessment model can then be tested by:

```shell
$ python test_room_layout.py -o <name> -c <checkpoint_filepath>
```
This will create a new figure named <name> in the directory 'roc_curves'. The figure shows the Receiver Operating Characteristic Curve for the Empty Rooms and Valid Rooms datasets. The Area Under the Curve is displayed in the figure legend.

See the files for details of the available options for the model.

## Visualise a Graph

A room can be visualised as a layout and its corresponding graph from its [data dictionary](https://github.com/deepmind/graph_nets) using:
```shell
$ visualise_room(data_dict)
$ plt.show()
```

## Room Generation

Unfortunately we cannot release the code to place an furniture object into the room or completely generate a layout as these require commercially senstive proprietary software.
