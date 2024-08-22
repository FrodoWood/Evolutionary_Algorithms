import random
import pandas
import numpy as np
import time
import copy

# file = pandas.read_csv("ulysses16.csv")
# file = pandas.read_csv("20cities.csv")
file = pandas.read_csv("cities22_12.csv")

n_points_csv = len(file)

# random.seed(42)
adjacency_matrix = [[0,20,42,35],
                    [20,0,30,34],
                    [42,30,0,12],
                    [35,34,12,0]]


def populate_adjacency_matrix(file):
    matrix = np.zeros(shape=(n_points_csv,n_points_csv))
    for i in file.index:
        for j in file.index:
            xi = file['x'][i]
            xj = file['x'][j]

            yi = file['y'][i]
            yj = file['y'][j]
            
            distance = np.sqrt(np.power(xi - xj, 2) + np.power(yi - yj, 2))
            matrix[i][j] = np.round(distance,2)
    return matrix



def evaluate_tsp(adjacency_matrix, route):
    cost = 0
    for x in range(n_points_csv):
        current_node = x
        next_node = (x+1) % len(route)
        cost += adjacency_matrix[route[current_node]][route[next_node]]
    return cost

def random_route(n_cities):
    random_list = random.sample(range(n_cities),n_cities)
    return random_list

def best_neighbour(adjacency_matrix, neighbourhood):
    lowest_cost = float('inf')
    shortest_route = []
    for route in neighbourhood:
        cost = evaluate_tsp(adjacency_matrix, route)
        if cost < lowest_cost:
            lowest_cost = cost
            shortest_route = route
    return shortest_route

# TASKS
adjacency_matrix_2 = populate_adjacency_matrix(file)

#Lab 3 tasks Evolutionary Algorithms
def generate_random_population(size):
    population = []
    for i in range(size):
        population.append(random_route(route_size))
    return population

def generate_cost_per_parent(population, adjacency_matrix):
    cost_parent = []
    for p in population:
        cost_parent.append(evaluate_tsp(adjacency_matrix,p))
    return cost_parent

# Variant 3
def generate_fitness_normalized(population, adjacency_matrix):
    fitness_normalized = []
    cost_parent = generate_cost_per_parent(population,adjacency_matrix)
    fitness_parent = np.array(cost_parent)
    fitness_parent = 1/fitness_parent
    fitness_parent = np.power(fitness_parent, 6)
    total_population_fitness = sum(fitness_parent)

    for fitness_value in fitness_parent:
        fitness_normalized.append(fitness_value/total_population_fitness)
    return np.array(fitness_normalized)

def tournament_selection(population, adjacency_matrix):
    # cost_per_parent = generate_cost_per_parent(population)
    #apply best neighbour twice to get 2 best parents as a pair
    pair = []
    population_copy = copy.deepcopy(population)
    for i in range(2):
        parent_A = population_copy[random.randint(0, len(population_copy) -1 )]
        parent_B = population_copy[random.randint(0, len(population_copy) -1 )]
        if evaluate_tsp(adjacency_matrix,parent_A) <= evaluate_tsp(adjacency_matrix, parent_B):
            pair.append(parent_A)
        else:
            pair.append(parent_B)
        
    # parent_1 = best_neighbour(adjacency_matrix,population_copy)
    # pair.append(parent_1)
    # population_copy.remove(parent_1)
    # parent_2 = best_neighbour(adjacency_matrix,population_copy)
    # pair.append(parent_2)
    # print(pair)
    return pair

# Variant 2
def roulette_selection(population, N_OFFSPRING, cumsum_fitness, adjacency_matrix):
    parents = []
    for _ in range(N_OFFSPRING):
        random_value = random.random()
        random_selection_value = random.random()
        prob_random_selection = 0.1
        if random_selection_value <= prob_random_selection:
            selected_index = random.randint(0,len(population) -1)
        else:
            selected_index = np.searchsorted(cumsum_fitness, random_value)
        # print(f'random value: {random_value}, selected index: {selected_index}')
        parents.append(population[selected_index])
    return parents


def select_parents(population, N_OFFSPRING, adjacency_matrix):
    parent_pairs = []
    population_copy = copy.deepcopy(population)
    for i in  range((int)(N_OFFSPRING/2)):
        pair = tournament_selection(population_copy, adjacency_matrix)
        parent_pairs.append(pair)
        # print(pair)
        # remove pair from population if I don't want the same parents to keep having kids over and over -> increase exploitation and decrease exploration
        # population_copy.remove(pair[0]) 
        # population_copy.remove(pair[1])
    return np.array(parent_pairs)

# Variant 1
def select_parents_roulette(population, N_OFFSPRING, adjacency_matrix):
    parent_pairs = []
    fitness_normalized = generate_fitness_normalized(population,adjacency_matrix)
    cumsum_fitness = np.cumsum(fitness_normalized)
    # get list of parents selected with roulette selection
    # make pairs of parents by randomly selecting and removing them from the list of parents 
    # add the pairs to the parent_pairs list
    parents_list = roulette_selection(population, N_OFFSPRING, cumsum_fitness, adjacency_matrix)
    while len(parent_pairs) < N_OFFSPRING/2:
        if len(parents_list) <= 1: break
        pair = []
        pair.append(parents_list.pop())
        pair.append(parents_list.pop())
        parent_pairs.append(pair)
    return np.array(parent_pairs)
    
def order_1_crossover(pair):
    children = []
    for i in range(2):
        parent_1 = pair[i%2]
        parent_2 = pair[(i+1)%2]

        parent_1_copy = copy.deepcopy(parent_1)
        parent_2_copy = copy.deepcopy(parent_2)
        parent_1_sub_list = []
        parent_2_sub_list = []
        child = []
        # print(f'first parent  {parent_1}')
        # print(f'second parent {parent_2}')
        sub_list_size = (int)(np.round(route_size/2))
        n1 = random.randint(0,route_size-1)
        n2 = (n1 + sub_list_size - 1) % route_size
        # print(f'n1: {n1} \nn2: {n2}')
        if n1 > n2:
            parent_1_sub_list = np.concatenate((parent_1[n1:], parent_1[:n2 + 1]))
        else:
            parent_1_sub_list.append(parent_1[n1:n2+1])
            parent_1_sub_list = parent_1_sub_list[0]
        # any element from the parent_1_sub_list that is also in parent_2 = -1 (used as a mask)
        parent_2_copy = np.where(np.isin(parent_2, parent_1_sub_list), -1, parent_2)
        # Getting the parent 2 sub list in order from the slice point i.e. n2 and reomving all -1 elements
        for i in range(route_size):
            parent_2_sub_list.append(parent_2_copy[(n2+i+1) % route_size])
            parent_2_sub_list = [x for x in parent_2_sub_list if x != -1]
        # Add the parent 2 sub list elements to the parent 1 (child 1) which already contains the parent 1 sub list
        for i in range(np.abs(len(parent_2) - len(parent_1_sub_list))):
            # print((n2+i+1) % route_size)
            child = parent_1_copy
            child[(n2+i+1) % route_size] = parent_2_sub_list[i]
        children.append(child)
    return np.array(children)

def swap_mutation(child):
    child_copy = copy.deepcopy(child)
    n1 = random.randint(0,len(child_copy)-1)
    n2 = (n1 + random.randint(1,len(child_copy) - 1)) % len(child_copy)
    temp = child_copy[n2]
    child_copy[n2] = child_copy[n1]
    child_copy[n1] = temp
    return child_copy

def evol_alg(population_size, N_OFFSPRING, recombination_probability, mutation_probability, adjacency_matrix, time_limit):
    population = generate_random_population(population_size)
    # print(f'fitness normalized: \n{np.array(generate_fitness_normalized(population, adjacency_matrix))}')

    all_time_best = []
    all_time_best_cost = float('inf')
    end_time = time.time() + time_limit
    # print(f'initial population is: {population}')
    while time.time() < end_time:
        parent_pairs = select_parents(population, N_OFFSPRING, adjacency_matrix)
        # print(f'parent pairs are: {parent_pairs}')
        children = []
        for pair in parent_pairs:
            prob = random.random()
            if prob <= recombination_probability:
                children_pair = order_1_crossover(pair)
                children.append(children_pair[0])
                children.append(children_pair[1])
                continue
            children.append(pair[0])
            children.append(pair[1])
        
        for child in children:
            prob = random.random()
            if prob <= mutation_probability:
                child = swap_mutation(child)
            else:
                continue

        population = np.array(children).tolist()
        #************** Only for debugging
        cost_parent = generate_cost_per_parent(population, adjacency_matrix)
        cost_parent = np.array(cost_parent)
        cost_parent = 1/cost_parent
        total_population_cost = sum(cost_parent)
        # print(f'Total population fitness tournament selection:{total_population_cost}')
        #************** Only for debugging
        # print(f'Children are: {population}')
        best_route = best_neighbour(adjacency_matrix,population)
        best_route_cost = evaluate_tsp(adjacency_matrix,best_route)
        if best_route_cost < all_time_best_cost:
            all_time_best = best_route
            all_time_best_cost = best_route_cost
            # population[-1] = all_time_best #elitism > adding the all time best to the population of children
            # print(all_time_best, all_time_best_cost)
        else:
            population[-1] = best_route #elitism > adding the all time best to the population of children


        # elitism -> always add the all time best to the population, add it to the end since the children of the best parents are in first few indeces of the population
        # if len(all_time_best) != 0:
        #     population[-1] = all_time_best
        # elitism for current best
        # population[-2] = best_route
        # print(population[-1])
    
    return all_time_best

def evol_alg_roulette(population_size, N_OFFSPRING, recombination_probability, mutation_probability, adjacency_matrix, time_limit):
    population = generate_random_population(population_size)
    # print(f'fitness normalized: \n{np.array(generate_fitness_normalized(population, adjacency_matrix))}')
    all_time_best = []
    all_time_best_cost = float('inf')
    start_time = time.time()
    end_time = time.time() + time_limit
    # print(f'initial population is: {population}')
    while time.time() < end_time:
        parent_pairs = select_parents_roulette(population, N_OFFSPRING, adjacency_matrix)
        # print(f'parent pairs are: {parent_pairs}')
        children = []
        for pair in parent_pairs:
            prob = random.random()
            if prob <= recombination_probability:
                children_pair = order_1_crossover(pair)
                children.append(children_pair[0])
                children.append(children_pair[1])
                continue
            children.append(pair[0])
            children.append(pair[1])
        
        for child in children:
            prob = random.random()
            if prob <= mutation_probability:
                child = swap_mutation(child)
            else:
                continue

        population = np.array(children).tolist()
        # print(f'Children are: {population}')
        best_route = best_neighbour(adjacency_matrix,population)
        best_route_cost = evaluate_tsp(adjacency_matrix,best_route)
        if best_route_cost < all_time_best_cost:
            all_time_best = best_route
            all_time_best_cost = best_route_cost
            print(f'New best at time {round(time.time()-start_time,2)}s: {all_time_best}, cost-> {all_time_best_cost}')
            population[-1] = all_time_best #elitism > adding the all time best to the population of children
            # print(all_time_best, all_time_best_cost)
        else:
            population[-1] = best_route #elitism > adding the all time best to the population of children
            continue
    
    return all_time_best

route_size = n_points_csv
time_limit = 10
population_size = 100
N_OFFSPRING = 100
recombination_prob = 1
mutation_prob = 0.3

# evol_alg_tournament_route = evol_alg(population_size, N_OFFSPRING, recombination_prob, mutation_prob, adjacency_matrix_2, time_limit)
# print(f'\nEvolutionary algorithm tournament after {time_limit} seconds:\n{evol_alg_tournament_route} cost: {evaluate_tsp(adjacency_matrix_2,evol_alg_tournament_route)}\n')

evol_alg_roulette_route = evol_alg_roulette(population_size, N_OFFSPRING, recombination_prob, mutation_prob, adjacency_matrix_2, time_limit)
print(f'\nEvolutionary algorithm roulette after {time_limit} seconds:\n {evol_alg_roulette_route} cost: {evaluate_tsp(adjacency_matrix_2,evol_alg_roulette_route)}\n')
