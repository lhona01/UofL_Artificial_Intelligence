#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt

#node class ------------------
class node:
    #node constructor
    def __init__(self, Id, x_coordinate, y_coordinate):
        self.Id = Id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.distanceToOtherNodes = []
    #End init
#End Node Class -------------------

#minimuCostPath Class ----------------
class minimumCostPath:
    def __init__(self, pathOrder, pathCost):
        self.pathOrder = pathOrder
        self.pathCost = pathCost
#End minimumCostPath
#Functions --------------
#distance function
def distance(x1, y1, x2, y2):
    d = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
    return d
#End distance

# find distance from a node to all other node
def oneToManyNodeDistance(currentNode, node_arr):
    for node in range(len(node_arr)):
        node_arr[currentNode].distanceToOtherNodes.append(distance(node_arr[currentNode].x_coordinate, node_arr[currentNode].y_coordinate,
                                     node_arr[node].x_coordinate, node_arr[node].y_coordinate))
#End oneToManyNodeDistance

#swap numbers in array
def swap(array, index1, index2):
    swap = array[index1]
    array[index1] = array[index2]
    array[index2] = swap
#End swap

#Generate all arangement for set of nodes (Permutation)/ recursive method
permutations = [] #store all permutations here
def permutation(setOfNode, startingPoint, level):
    if (startingPoint == level):
        permutations.append(setOfNode.copy())
    else:
    
        for node in range(startingPoint, level):
            swap(setOfNode, startingPoint, node)
            permutation(setOfNode, startingPoint + 1, level)
            swap(setOfNode, node, startingPoint)
#End Generate Permutation

#Finds and returns minimum cost route
def minimumCost(permutations, node_arr):
    allShortestPaths = []
    allPathDistances = []
    startingAndEndingNode = permutations[0][0]
    minDistance = -1
    for path in permutations:
        compare = 0
        startingAndEndingNode = path[0]
        for node in range(len(path)):
            if node + 1 < len(path):
                compare += node_arr[path[node]].distanceToOtherNodes[path[int(node) + 1]]
                #print(path[node], " ", path[int(node)+1], " ", node_arr[path[node]].distanceToOtherNodes[path[int(node) + 1]])
            else:
                compare += node_arr[path[node]].distanceToOtherNodes[startingAndEndingNode]
                #print(path[node], " ", startingAndEndingNode, " ", node_arr[path[node]].distanceToOtherNodes[startingAndEndingNode])
        #print(compare)
        allPathDistances.append(compare)
        #initialize minimum disttance/cost
        if minDistance == -1:
            minDistance = compare
        elif minDistance > compare:
            minDistance = compare
    #Find and return path with least distance
    for distance in range(len(allPathDistances)):
        if allPathDistances[distance] == minDistance:
            allShortestPaths.append(minimumCostPath(list(permutations[distance]), allPathDistances[distance]))
    return allShortestPaths
#End minimumCostPath
#-------------------------------

#Begin Main Code
#open file
try:
    file = open('Random4.tsp', 'r')
except FileNotFoundError:
    print('The file was not found!')

#read lines with node coordinates
lines = np.array(file.readlines())[7:]
node_arr = []
setOfNode = []
for line in range(len(lines)):
    x = lines[line].split()
    #node starts from 0 instead of 1
    node_arr.append(node(int(x[0]) - 1, float(x[1]), float(x[2])))
    setOfNode.append(int(x[0]) - 1)

#assign distance from one node to other nodes for all nodes
for node in range(len(node_arr)):
    oneToManyNodeDistance(node, node_arr)

#permutation function call (all possible permutation stored in list named permutations)
permutation(setOfNode, 0, len(setOfNode))


#plot node coordinates in 2d graph
# xpoints = []
# ypoints = []
# for node in node_arr:
#     xpoints.append(node.x_coordinate)
#     ypoints.append(node.y_coordinate)
# plt.scatter(xpoints, ypoints)
# plt.show()

#print coordinates of each coordinates
for node in node_arr:
    print('node:', node.Id, '(', node.x_coordinate, ',', node.y_coordinate, ')')

print('\nshortest paths:')
#return all shortest paths
allShortestPaths = minimumCost(permutations, node_arr)
for path in allShortestPaths:
    path.pathOrder.append(path.pathOrder[0])
    print(path.pathCost, path.pathOrder)
#End Main Code


# In[ ]:




