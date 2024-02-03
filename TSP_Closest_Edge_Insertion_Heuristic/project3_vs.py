import numpy as np
import matplotlib.pyplot as plt
import math
import time

#node class -----------------------------------------
class node:
    #node constructor
    def __init__(self, Id, x_coordinate, y_coordinate):
        self.Id = Id
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
    #End Class node

#slope
def slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)
#End slope

#y-intercept
def y_intercept(m, x, y):
    return y - m * x
#End y_intercept

#distance between two points
def distance(x1, y1, x2, y2):
    d = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
    return d
#End distance

#Perpendicular distance
#Is node within perpendicular range of the edge
#yes, find perpendicular distance
#no, find nearest visited node
def edgeOrNodeDistance(x1, y1, x2, y2, node, visitedNodes):
    m = slope(x1, y1, x2, y2) #slope of edge
    b = y_intercept(m, x1, y1) #y intercept of edge
    m1 = -1/m # perpendicular slope
    b1 = y_intercept(m1, node.x_coordinate, node.y_coordinate)
    b2 = y_intercept(m1, x1, y1) # y intercept
    b3 = y_intercept(m1, x2, y2) # y intercept

    if m1 > 0:
        if x1 < x2:
            a2 = m1 * node.x_coordinate + b2 # y = mx + b
            a3 = m1 * node.x_coordinate + b3 # y = mx + b
        else:
            a2 = m1 * node.x_coordinate + b3 # y = mx + b
            a3 = m1 * node.x_coordinate + b2 # y = mx + b
    else:
        if x1 > x2:
            a2 = m1 * node.x_coordinate + b2 # y = mx + b
            a3 = m1 * node.x_coordinate + b3 # y = mx + b
        else:
            a2 = m1 * node.x_coordinate + b3 # y = mx + b
            a3 = m1 * node.x_coordinate + b2 # y = mx + b
    
    if node.y_coordinate <= a2 and node.y_coordinate >= a3: # inbetween peprpendicular line of two connected nodes
        # find perpendicular distance of the node
        x = (b1 - b) / (m - m1)# x_coordinate where perpendicular line intersects
        y = m * x + b # y_coordinate where perpendicular line intersects
        return distance(node.x_coordinate, node.y_coordinate, x, y)
    else:
        distanceToNode = -1
        #for point in visitedNodes:
        d = distance(node.x_coordinate, node.y_coordinate, x1, y1)
        if distanceToNode == -1:
            distanceToNode = d
        else:
            if d < distanceToNode:
                distanceToNode = d
    return distanceToNode
#End edgeOrNodeDistance

#Begin Main
visitedNodes = []
#open file
try:
    file = open('Documents/Artificial Intelligence/Project3/Random40.tsp', 'r')
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

#find very first node and last node (x_coordinate) in a graph (Starting Edge)
x_min_node = 0
x_max_node = 0
x_min_value = 10000
x_max_value = -1
for i in range(len(node_arr)):
    if node_arr[i].x_coordinate < x_min_value:
        x_min_value = node_arr[i].x_coordinate
        x_min_node = i
    if node_arr[i].x_coordinate > x_max_value:
        x_max_value = node_arr[i].x_coordinate
        x_max_node = i
visitedNodes.append(node_arr[x_min_node])
visitedNodes.append(node_arr[x_max_node])
visitedNodes.append(visitedNodes[0])

# find the nearest node (starting from the sstart edge)
elapsedTime = 0
startTime = time.time()

z = 0
while z < len(node_arr) - 2:
    nearestEdgeNode =- -1
    nodeEdgeDistance = -1
    visitedNodeInsertIndex = -1
    for i in range(len(visitedNodes)- 1):
        for x in range(len(node_arr)):
            if node_arr[x] not in visitedNodes:
                if node_arr[x].Id == 17 and visitedNodes[i].Id == 11:
                    dist = edgeOrNodeDistance(visitedNodes[i].x_coordinate, visitedNodes[i].y_coordinate, visitedNodes[i + 1].x_coordinate, visitedNodes[i + 1].y_coordinate, node_arr[x], visitedNodes)
                else:
                    dist = edgeOrNodeDistance(visitedNodes[i].x_coordinate, visitedNodes[i].y_coordinate, visitedNodes[i + 1].x_coordinate, visitedNodes[i + 1].y_coordinate, node_arr[x], visitedNodes)

                #if (dist == 3.9410083998514955):
                    #print("outside", visitedNodes[i].Id, dist)

                if nodeEdgeDistance == -1 and dist != -1:
                    nearestEdgeNode = x
                    nodeEdgeDistance = dist
                    visitedNodeInsertIndex = i + 1
                elif dist < nodeEdgeDistance and dist != -1:
                    nearestEdgeNode = x
                    nodeEdgeDistance = dist
                    visitedNodeInsertIndex = i + 1
    if nearestEdgeNode != -1 and visitedNodeInsertIndex != -1:
        visitedNodes.insert(visitedNodeInsertIndex, node_arr[nearestEdgeNode])
    
    z = z + 1

endTime = time.time()
elapsedTime = endTime - startTime

#Calculate the total distance
sumDistance = 0
for i in range(len(visitedNodes)- 1):
    sumDistance = sumDistance + distance(visitedNodes[i].x_coordinate, visitedNodes[i].y_coordinate, visitedNodes[i+1].x_coordinate, visitedNodes[i+1].y_coordinate)

apple = []
for i in visitedNodes:
    apple.append(i.Id + 1)     
print(apple)

# scatter plot all points
x = []
y = []
for i in range(len(node_arr)):
    x.append(node_arr[i].x_coordinate)
    y.append(node_arr[i].y_coordinate)
    
for i in range(len(node_arr)):
    plt.scatter(node_arr[i].x_coordinate, node_arr[i].y_coordinate)
    plt.text(node_arr[i].x_coordinate, node_arr[i].y_coordinate, node_arr[i].Id + 1)

x_points = []
y_points = []
plt.title("Cost:" + str(sumDistance) + ", Path:" + str(apple) + ", Time:" + str(elapsedTime))
for i in range(len(visitedNodes)):
    x_points.append(visitedNodes[i].x_coordinate)
    y_points.append(visitedNodes[i].y_coordinate)
    plt.plot(x_points, y_points)
    plt.pause(0.001)
plt.show()
plt.show()
