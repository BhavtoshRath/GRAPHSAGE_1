import re
import json

test_com_set = set()
val_com_set = set()
test_nodes = set()
val_nodes = set()
test_com_set.add(4)
val_com_set.add(1)
val_com_set.add(3)
with open('data/com_NBC_l1_sample.txt') as infile:
    for line in infile:
        dict_NBC = json.loads(line)
        print(dict_NBC)
        test_nodes = test_nodes.union(set([node for node in dict_NBC['boundary'] if dict_NBC['cl'] in test_com_set]))
        val_nodes = val_nodes.union(set([node for node in dict_NBC['boundary'] if dict_NBC['cl'] in val_com_set]))

test_nodes = test_nodes.difference(val_nodes)
print(test_nodes)
print(val_nodes)
