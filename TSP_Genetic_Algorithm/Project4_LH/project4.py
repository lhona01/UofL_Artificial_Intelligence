import numpy as np
import copy
import math
import random
import matplotlib.pyplot as plt
import tkinter as tk

class node:
    #nodes constructor
    def __init__(self, id, x, y):
        self.Id = id
        self.x = x
        self.y = y
#End class nodes

class individual:
    def __init__(self, path, path_distance, fitness):
        self.path = path
        self.path_distance = path_distance
        self.fitness = fitness #fitness compared to other indivuduals in population 
#End class individual

class chromosome:
    def __init__(self, chromosome, index):
        self.selected_chromosome = chromosome
        self.index = index #used for sorting, index of parents chromosome that got selected
#End class chromosome

#distance between two points
def distance(x1, y1, x2, y2):
    d = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
    return d
#End distance

#initialize individual (initial random path)
def initializeIndividual(node_id):
    new_path = copy.deepcopy(node_id)
    np.random.shuffle(new_path)
    return individual(new_path, 0, 0)
#End initialize individual

#Path Distance (total distance of a path)
def calcPathDistance(path, node_arr):
    path_distance = 0
    closed_path = copy.deepcopy(path)
    closed_path.append(path[0])
    for f in range(len(path) - 1):
        path_distance += distance(node_arr[closed_path[f]].x, node_arr[closed_path[f]].y, node_arr[closed_path[f+1]].x, node_arr[closed_path[f+1]].y)
    return path_distance
#End path distance

#Path fitness (fitness compared to other individuals in population)
def pathFitness(path_distance, sum_fitness):
    fitness = path_distance / sum_fitness
    return fitness
#End path fitness
    
#Roulette wheel selection (parent selection)
def rouletteWheel(roulette_pie):
    pie_length = (len(roulette_pie)) # include all parents into a pool of a roulette wheel
    spin_result = random.uniform(0, 1) # spin the roulette wheel
    for f in range(pie_length - 1): 
        if  0 <= spin_result and spin_result < roulette_pie[0]: 
            return f # return index
        elif roulette_pie[f+1] > spin_result and spin_result >= roulette_pie[f]: 
            return f # return index
#End roulette wheel

#Tournament selection
def tournamentSelection(individuals, tournament_size):
    selected = []
    while len(selected) < 2:
        best_individual = []
        tournament = random.sample(individuals, tournament_size)
        best_individual.append(min(tournament, key=lambda r: r.fitness))
        selected.append(best_individual[0].path)
    return selected

#ordered crossver function (produces 2 children)
def orderedCrossover(parent1, parent2):
    length = len(parent1) #length of the parents
    child = [None] * length

    index1, index2 = random.sample(range(length), 2)
    start = min(index1, index2)
    end = max(index1, index2)

    child[start:end + 1] = parent1[start:end + 1]
    remaining_value = [node for node in parent2 if node not in child]

    for i in range(length):
        if child[i] is None:
            child[i] = remaining_value.pop(0)
    return child
#End randomCrossover

#cyclic crossover
def cycleCrossover(parent1, parent2):
    length = len(parent1)
    child = [-1] * length
    cycle = []

    while -1 in child:
        index = child.index(-1)
        current_parent = parent1 if index % 2 == 0 else parent2

        while index not in cycle:
            cycle.append(index)
            index = parent1.index(parent2[index])

        for i in cycle:
            child[i] = current_parent[i]

        cycle = []

    return child
#End cycle crossover

#Mutation Swap two random cities
def mutation(path):
    index1, index2 = random.sample(range(len(path)), 2)
    path[index1], path[index2] = path[index2], path[index1]
#End mutation

#main function
def main():
    #open file
    try:
        file = open('Random100.tsp', 'r')
    except FileNotFoundError:
        print('The file was not found!')
    
    read_line = np.array(file.readlines())[7:]
    node_arr = []
    node_id = []
    for line in range(len(read_line)):
        x = read_line[line].split()
        #node starts from 0 instead of 1
        node_arr.append(node(int(x[0]) - 1, float(x[1]), float(x[2])))
        node_id.append(line)

    #Constant mutation, crossover, random chromosome rate
    CROSSOVER_PERCENTAGE = 98
    MUTATION_PERCENTAGE = 100 - CROSSOVER_PERCENTAGE
    TOURNAMENT_SIZE = 10

    #Create starting population of 10
    populationSize = 100 # fixed population size for the program
    individuals = [] # individuals of the population
    roulette_pie = []   # roulete wheel with all individuals fitness
    sum_fitness = 0 # Total fitness of all individuals combined
    sum_roulette_pie = 0 # should be same a sum_fitness
    iter_generation = 500 # number of generation to produce
    generation = 0 #current generation
    children = [] #current generation
    best_distance_per_gen = [] #best path distance for every generation
    best_path_per_gen = [] #best path for every generation

    #Initializing all features of each path/individual
    #Random Paths (Starting point/Initialize individuals)
    for i in range(len(node_id)):
        individuals.append(initializeIndividual(node_id))
    #Path distance
    for i in range(len(individuals)):
        individuals[i].path_distance = calcPathDistance(individuals[i].path, node_arr)
        sum_fitness += individuals[i].path_distance
    #Path fitness
    for i in range(len(individuals)):
        individuals[i].fitness = pathFitness(individuals[i].path_distance, sum_fitness)
    
    best_distance_per_gen.append(min(individuals, key=lambda r: r.path_distance).path_distance) #min path distance from instances of class individuals for 0th generation
    #End Initializing all features of each path/individual

    while generation < iter_generation: #number of generation to produce
        sum_fitness = 0
        children = []
        best_path = []
        best_distance = None
        while len(children) < populationSize: #number of children to produce for every generation
            #Divide roulete wheel based on fitness
            sum_roulette_pie = 0
            for i in range(len(individuals)):
                if len(roulette_pie) == 0:
                    roulette_pie.append(individuals[i].fitness)
                    sum_roulette_pie = individuals[i].fitness
                else:
                    roulette_pie.append(sum_roulette_pie + individuals[i].fitness)
                    sum_roulette_pie += individuals[i].fitness
            #End Divide roulete wheel based on fitness

        #--------------UnComent For RouletteWheel Selection------------------------------------------
            #parent1 = individuals[rouletteWheel(roulette_pie)].path
            #parent2 = individuals[rouletteWheel(roulette_pie)].path
        #---------------------Uncomment for TOurnament Selection---------------------------------------
            parents = tournamentSelection(individuals, TOURNAMENT_SIZE)
            parent1 = parents[0]
            parent2 = parents[1]
        #--------------------UnComment for Ordered Crossover----------------------------------------
            child = orderedCrossover(parent1, parent2) #offspring path of parent1 and parent 2
        #-------------------Uncomment for cycle crossover------------------------------------
            #child = cycleCrossover(parent1, parent2)
            if random.uniform(0, len(child)) < MUTATION_PERCENTAGE:
                mutation(child)
            path_distance = calcPathDistance(child, node_arr) #childs path distance
            #replace shorter path
            if best_distance == None:
                best_distance = path_distance
                best_path = child
            elif path_distance < best_distance:
                best_distance = path_distance
                best_path = child
            sum_fitness += path_distance 
            children.append(individual(child, calcPathDistance(child, node_arr), 0))
            #import collections
            #print([item for item, count in collections.Counter(children[0]).items() if count > 1])
        generation += 1
        
        for i in range(len(children)): #calculate childrens fitness
            children[i].fitness = pathFitness(children[i].path_distance, sum_fitness)
        
        individuals = copy.deepcopy(children) #copy new childrens and let go of previous generation
        best_distance_per_gen.append(min(individuals, key=lambda r: r.path_distance).path_distance) #min path distance from instances of class individuals for 0th generation 
        best_path_per_gen.append(best_path)
    
    x_axis= []
    for i in range(len(best_distance_per_gen)):
        x_axis.append(i)
    for i in best_distance_per_gen:
        plt.plot(x_axis, best_distance_per_gen)
        plt.ylim(0,)
        plt.title("Ordered crossover with Tournament Selection\n500 Generation")
    plt.show()

    fig, ax = plt.subplots()
    print(best_distance_per_gen[0], best_distance_per_gen[-1])
    for i in range(len(best_path_per_gen)):
        x, y = [], []
        title = 'Ordered crossover with Tournament Selection' + '\nGeneration: ' + str(i) + ' distance: ' + str(best_distance_per_gen[i])
        for j in best_path_per_gen[i]:
            x.append(node_arr[j].x)
            y.append(node_arr[j].y)
        ax.plot(x, y, marker='o')
        ax.set_title(title)
        plt.pause(0.01)
        if i < len(best_path_per_gen) - 1:
            ax.clear()

    plt.show()

main()