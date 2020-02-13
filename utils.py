from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import re

import networkx as nx

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11" #br-1

WALK_LEN=5
N_WALKS=50


def load_data(normalize=True):
    # Read netwrok file
    network_file = 'data/network_sample.txt'
    G = nx.DiGraph()
    with open(network_file) as infile:
        for line in infile:
            l_spl = re.split(',', line.rstrip())
            G.add_edge(int(l_spl[0]), int(l_spl[1]))

    # Read spreaders file
    spreaderFile = 'data/spreaders.txt'
    spreaderSet = set()
    with open(spreaderFile) as infile:
        for line in infile:
            spreaderSet.add(int(line.rstrip()))

    train_nodes = set()
    test_nodes = set()
    val_nodes = set()
    boundary_nodes = set()
    # Read nodes (train/test/val split)
    comFile = 'data/com_NBC_l1_sample.txt'
    with open(comFile) as infile:
        for line in infile:
            dict_NBC = json.loads(line)
            boundary_nodes = boundary_nodes.union(set([node for node in dict_NBC['boundary']]))
            train_nodes = set(random.sample(boundary_nodes, int(0.6 * (len(boundary_nodes)))))
            test_nodes = set(random.sample(boundary_nodes.difference(train_nodes), int(0.2 * (len(boundary_nodes)))))
            val_nodes = boundary_nodes.difference(train_nodes.union(test_nodes))

    featureFile = 'data/features.txt'
    labels = {}
    feats = np.loadtxt(featureFile, delimiter=",")
    with open(featureFile) as infile:
        for line in infile:
            l_spl = re.split(',', line.rstrip())
            # feats[int(l_spl[0])] = np.array(l_spl[1:], dtype="float64")
            if int(l_spl[0]) in spreaderSet:
                labels[int(l_spl[0])] = [1, 0]
            else:
                labels[int(l_spl[0])] = [0, 1]

    # Normalize features
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([1, 3, 4])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return G, feats, labels, train_nodes, test_nodes, val_nodes

