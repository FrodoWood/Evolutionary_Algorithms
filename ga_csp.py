import random
import numpy as np
import time
import copy
import math
import csv
# stocks = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
# stock_price = [86, 85, 83, 79, 68, 66, 64, 63]
# order_len = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# order_q = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

stocks = [120, 115, 110, 105, 100]
stock_price = [12, 11.5, 11, 10.5, 10]
order_len = [21, 22, 24, 25, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63, 65, 66, 67]
order_q = [13, 15, 7, 5, 9, 9, 3, 15, 18, 17, 4, 17, 20, 9, 4, 19, 4, 12, 15, 3, 20, 14, 15, 6, 4, 7, 5, 19, 19, 6, 3, 7, 20, 5, 10, 17]

# stocks = [10,13,15]
# stock_price = [100,130,150]
# order_len = [3,4,5,6,7,8,9,10]
# order_q = [5,2,1,2,4,2,1,3]

# stocks = [50,80,100]
# stock_price = [100,175,250]
# order_len = [20,25,30]
# order_q = [5,7,5]



def random_solution(order_lengths, order_q, n):
    # ORIGINAL produces zeros towards the end of the rows
    solution = np.zeros(shape=(len(order_lengths), n))
    for index, order_len in enumerate(order_lengths):
        quantity = order_q[index]
        quantity_left  = quantity
        activity = np.zeros(shape=(n))
        # print(activity)
        for i in range(n):
            random_q = random.randint(0,quantity_left)
            if sum(activity) == quantity:
                break
            if(i == n-1):
                activity[i] = quantity_left
                break
            activity[i] = random_q
            quantity_left -= random_q
        np.random.shuffle(activity)
        solution[index] = activity

    return solution

def random_solution_2(order_lengths, order_q, n):
    # SECOND 
    solution = np.zeros(shape=(len(order_lengths), n))
    for index, order_len in enumerate(order_lengths):
        q = order_q[index]
        segments = np.random.randint(0, q, size = n )
        segments.sort()

        diff = np.diff(segments)
        row = np.append(diff, q - segments[-1])
        error = q - sum(row)
        selected_index = random.randint(0,len(row) -1)
        row[selected_index] += error

        np.random.shuffle(row)

        if sum(row) != q:
            print('Sum of row is not equal to Q!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        solution[index] = row
        # print(row)
    return solution        
            
def evaluate_csp_old(solution, stocks, stock_price, order_len, order_q):
    sol_by_stock_len = np.transpose(solution)
    cost = np.zeros(shape= sol_by_stock_len.shape[0])
    for index, stock_len in enumerate(sol_by_stock_len):
        # Calculate the total length of stock needed for each order length
        total_stock_order = stock_len * order_len
        # Calculate the number of stock lengths needed in total for all order lengths, and then multiply that by the stock price.
        total_stock_len_price = math.ceil(sum(total_stock_order) / stocks[index]) * stock_price[index]
        cost[index] = total_stock_len_price
    return sum(cost)

def evaluate_csp(solution, stocks, stock_price, order_len, order_q):
    # activities
    activities_all = solution_to_activities(solution, stocks, stock_price,order_len,order_q)
    # for each stock get the value from activities using stock as key such as activities.get(stocks[0]) -> activities for L50
    # len of activities list for each stock is the number of stock lengths needed, multiply that by cost to get total cost for each stock
    # add up all the costs
    stock_price_dict = dict(zip(stocks, stock_price))
    cost = 0
    for key in activities_all:
        activities = activities_all[key]
        # print(f'key :{key}')
        # print(activities)
        cost += len(activities) * stock_price_dict[key]
    return cost

def solution_to_activities(solution, stocks, stock_price, order_len, order_q):
    sol = np.transpose(solution).astype(int)
    dict1 = {}
    for i, stock in enumerate(stocks):
        my_list = []
        for j, order in enumerate(order_len):
            my_list.extend([order] * sol[i][j])
        # my_list.reverse()
        my_list.sort(reverse=True)
        # print(my_list)
        dict1[stock] = my_list
    # print(solution)
    # print(dict1, '\n')
    dict2 = {}

    for stock in stocks:
        orders = dict1[stock]
        if len(orders) == 0:
            continue
        activities = [[]]
        index = 0
        while len(orders) > 0:
            order = orders[index]
            # If the current order len + last activity's sum is exactly equal to stock then 
            # it means that I should stop iterating thorugh the orders to check for another order len to add.
            # It also means that I can add en empty activity so that next iteration I don't have to check that same activity again.
            if sum(activities[-1]) + order == stock:
                activities[-1].append(order)
                orders.remove(orders[index])
                activity = []
                if len(orders) > 0:
                    activities.append(activity)
                index = 0
                continue
            # I know that current order + sum of last activity is not equal to stock length, so check if it's
            # less thank stock length, this means that more orders can be added.
            if sum(activities[-1]) + order < stock:
                activities[-1].append(order)
                orders.remove(orders[index])
                index = 0
                continue
            # The current order len doesn't add up to exactly the stock len and goes over the stock length if added to activity
            else: # sum of last activity + order > stock
                # Continue checking next order len if there's still orders left that maybe could fit until the end of the list
                if index != (len(orders)-1):
                    index += 1
                    continue # This doens't work because it doesn't add the original order to a new activity but the one on the current iteration. I need to find a way to remember what the original length was 
                else:
                    # Have gone through the entire list without a valid order to add, so add the first order from which I started
                    activity = [orders[0]]
                    orders.remove(orders[0])
                    activities.append(activity)
                    index = 0
                    continue
        dict2[stock] = activities
    # print(dict2)
    return dict2

def random_csp(stocks, stock_price, order_len, order_q, time_limit):
    end_time = time.time() + time_limit
    best_sol = []
    best_sol_cost = float('inf')
    iterations = 0
    while time.time() < end_time: 
        # Old evaluate csp based cost calculation
        # solution = random_solution(order_len,order_q, len(stocks))      
        # cost = evaluate_csp_old(solution, stocks, stock_price, order_len)
        
        solution = random_solution(order_len,order_q, len(stocks))
        # activities = solution_to_activities(solution, stocks,stock_price, order_len, order_q )
        cost = evaluate_csp(solution, stocks, stock_price, order_len, order_q)
        # If the cost is lower than best cost then make s1 the best sol
        if(cost < best_sol_cost):
            best_sol = solution
            best_sol_cost = cost
            print(f'New best found: {best_sol_cost}')
        iterations += 1
    print(f'Iterations : {iterations}')
    return best_sol, best_sol_cost

def best_solution(population, stocks, stock_price, order_len, order_q):
    cost = []
    for person in population:
        cost.append(evaluate_csp(person, stocks, stock_price, order_len, order_q))

def generate_random_population(size, stocks, order_len, order_q):
    population = []
    for _ in range(size):
        population.append(random_solution_2(order_len,order_q,len(stocks)))
    return np.array(population)

def generate_population_cost(population, stocks, stock_price, order_len, order_q ):
    population_cost = []
    for person in population:
        population_cost.append(evaluate_csp(person,stocks, stock_price, order_len, order_q ))
    return population_cost

def roulette_selection(population, pop_cost, pop_size, exponent, stocks, stock_price, order_len, order_q):
    pop_fitness = np.array(pop_cost)
    # average_cost = sum(pop_fitness)/len(population)
    pop_fitness = 1/pop_fitness
    pop_fitness = np.power(pop_fitness, exponent)
    # pop_fitness = np.log2(pop_fitness)
    norm_pop_fitness = pop_fitness / sum(pop_fitness)
    cumsum_fitness = np.cumsum(norm_pop_fitness)

    return_pop = []
    return_cost = []
    list_of_selected_indeces = []

    while len(return_pop) < pop_size:
        r = random.random()
        selected_index = np.searchsorted(cumsum_fitness, r)
        # print(selected_index)
        # print(f'average cost: {average_cost}')
        # print(pop_cost[selected_index])
        # if selected_index in list_of_selected_indeces:
        #     continue
        list_of_selected_indeces.append(selected_index)
        selected_sol = population[selected_index]
        selected_sol_cost = pop_cost[selected_index]

        return_pop.append(selected_sol)
        return_cost.append(selected_sol_cost)

    # print(return_pop[0])
    return np.array(return_pop), np.array(return_cost)

def crossover(s1, s2):
    # part 1 of s1
    # part 2 of s2
    # -------------
    # part 1 of s2
    # part 2 of s1
    cut = random.randint(1,len(s1)-1)
    # print(f'cut {cut}')
    s1_p1 = s1[:cut] 
    s2_p2 = s2[cut:]
    s1s2 = np.vstack((s1_p1,s2_p2))
    s1_p2 = s1[cut:]
    s2_p1 = s2[:cut]
    s2s1 = np.vstack((s2_p1,s1_p2))
    # print(f's1 + s2:\n{s1s2}\n')
    # print(f's2 + s1:\n{s2s1}')
    return s1s2, s2s1

def crossover_uniform(s1,s2):
    rows, cols = s1.shape
    child = np.zeros((rows, cols))

    for i in range(rows):
        if random.random() < 0.5:
            parent = s1
        else:
            parent = s2
        child[i,:] = parent[i,:]
    return child

def global_recombination(solution, population):
    result = np.zeros_like(solution)  # Use np.zeros_like to create an array with the same shape as solution
    _, n_cols = solution.shape
    choices = np.arange(len(population))
    random_index = np.random.choice(choices)
    random_sol = population[random_index]
    for j in range(n_cols):
        result[:, j] = random_sol[:, j]
    return result


def mutation(solution):
    solution = np.array(solution)
    rows, cols = solution.shape
    # Increasing random element by random n
    indeces = np.arange(cols)
    i = np.random.choice(indeces)
    random_n = 20 # Quantity value to add/subtract

    for row_index, row in enumerate(solution):
        if random.random() < 0.1:
            continue
        row[i] += random_n

        desired_sum = order_q[row_index]

        row = row/sum(row) * desired_sum
        rounded_row= np.round(row).astype(int)

        # ERROR correction
        error = desired_sum - sum(rounded_row)
        rounded_row[i] += error
        if sum(rounded_row) != desired_sum:
            print(f'Messed up sum q: {row_index}')
        solution[row_index] = rounded_row

    # # Local search
    # solution = local_search(solution)
        
    return solution

def large_mutation(solution):
    solution = np.array(solution)
    total_rows = len(solution)
    for _ in range(random.randint(1, len(solution) -1 )): # Pick how many rows to mutate
        row_index = random.randint(0, total_rows - 1) # Pick a row to mutate
        row = solution[row_index]
        # Increasing random element by random n
        indeces = np.where(row >= 0)[0]
        i = np.random.choice(indeces)
        # print(f'index of element getting decremented: {i}')
        random_n = random.randint(1,max(order_q)) # Quantity value to add/subtract
        row[i] += random_n

        # Decreasing random element that is bigger than random n
        indeces = np.where(row >= random_n)[0]
        if len(indeces) > 0:
            i = np.random.choice(indeces)
            # print(f'index of element getting incremented: {i}\n')
        else: print('--------------------')
        row[i] -= random_n
        solution[row_index] = row
        
    return solution

def local_search(solution):
    improved_solution = np.copy(solution)  # Create a copy to avoid modifying the original

    # Iterate through each element in the solution matrix
    for i in range(improved_solution.shape[0]):
        for j in range(improved_solution.shape[1]):
            # Try decrementing the current quantity
            if improved_solution[i, j] > 0:
                decreased_quantity = 1
                increased_quantity = 1

                # Adjust another element in the row
                # while (j_increased := np.random.choice(improved_solution.shape[1])) == j:
                #     pass
                # j_increased = (j+1) % improved_solution.shape[1]
                j_increased = np.random.choice(improved_solution.shape[1])
                # Decrement the current and increment the adjusted
                improved_solution[i, j] -= decreased_quantity
                improved_solution[i, j_increased] += increased_quantity

                # Check if the new solution is better than the current one
                if evaluate_csp(improved_solution, stocks, stock_price, order_len, order_q) < evaluate_csp(solution, stocks, stock_price, order_len, order_q):
                    solution = np.copy(improved_solution)
                    return solution
                else:
                    # If not, revert the change
                    improved_solution[i, j] += decreased_quantity
                    improved_solution[i, j_increased] -= increased_quantity

    return solution

def column_mutation(solution):
    solution = np.array(solution)
    n_rows, n_cols = solution.shape

    # Get 2 random colums to swap
    random_indices_col = np.arange(n_cols)
    np.random.shuffle(random_indices_col)
    random_indices_col = random_indices_col[:2]
    col_index_1 = random_indices_col[0]
    col_index_2 = random_indices_col[1]

    # Get random row to apply mutation
    # row_index = random.randint(0,len(solution)-1)

    solution[:, [col_index_1, col_index_2]] = solution[:, [col_index_2, col_index_1]]
    # print()
    # print(solution)
    return solution

def row_mutation(solution):
    solution = np.array(solution)
    n_rows, n_cols = solution.shape

    # Get random row to apply mutation
    row_index = random.randint(0,len(solution)-1)
    row = solution[row_index]
    # s1 = random_solution(order_len, order_q, len(stocks))
    random_numbers = np.arange(n_cols)
    np.random.shuffle(random_numbers)
    n1 = random_numbers[0]
    n2 = random_numbers[1]

    row[n1], row[n2] = row[n2], row[n1]
    
    # solution[row_index] = s1[row_index]
    solution[row_index] = row

    # print()
    # print(solution)
    return solution

def tournament_selection(tournament_size, population, pop_cost, pop_size, stocks, stock_price, order_len, order_q):
    population = np.array(population)
    pop_cost = np.array(pop_cost)
    new_pop = []
    new_pop_cost = []
    while len(new_pop) < pop_size:
        participants = []
        cost = []
        for _ in range(tournament_size):
            selected_index = random.randint(0,len(population)-1)
            participant = population[selected_index]
            participants.append(participant)
            cost.append(pop_cost[selected_index])

        best = min(zip(participants,cost), key= lambda x: x[1]) # Tuple of participant and cost
        new_pop.append(best[0]) # Adding only the participant to the population
        new_pop_cost.append(best[1]) # Adding only the participant to the population


    return np.array(new_pop), np.array(new_pop_cost)

def random_selection(population, pop_cost, pop_size):
    new_pop = []
    new_pop_cost = []
    while len(new_pop) < pop_size:
        selected_index = random.randint(0,len(population)-1)
        new_pop.append(population[selected_index])
        new_pop_cost.append(pop_cost[selected_index])
    return np.array(new_pop), np.array(new_pop_cost)


def ga_csp_base(pop_size, mutation_prob, tournament_size, stocks, stock_price, order_len, order_q, time_limit):
    population = generate_random_population(pop_size, stocks, order_len, order_q)
    pop_cost = generate_population_cost(population,stocks, stock_price,order_len,order_q)
    best = []
    best_cost = float('inf')
    generations = 0
    start_time = time.time()
    end_time = time.time() + time_limit
    
    # Data export
    data = []

    while time.time() < end_time:
        elapsed_time = time.time() - start_time
        gen_data = []

        tournament_result = tournament_selection(tournament_size,population, pop_cost, pop_size,stocks, stock_price, order_len, order_q)
        parents = tournament_result[0]
        parents_cost = tournament_result[1]

        mutated_children = []
        cost_children = []
        for parent in parents:
            n = random.random()
            if n < mutation_prob:
                mutated_child = row_mutation(parent)
            else: 
                mutated_child = parent
            mutated_children.append(mutated_child)
        cost_children = generate_population_cost(mutated_children, stocks, stock_price, order_len, order_q)

        best_child = min(zip(mutated_children,cost_children), key= lambda x: x[1]) # Tuple of participant and cost
        # if the best cost in this batch of children is better than global best -> set global best as best
        if best_child[1] < best_cost:
            best = best_child[0]
            best_cost = best_child[1]
            print(f'new best found at time {round(time.time() - start_time)}: {best_cost}')
            
            # --------------Record data to export----------------
            gen_data.append(generations)
            gen_data.append(elapsed_time)
            gen_data.append(best_cost)
            gen_data.append(len(population))
            gen_data.append(mutation_prob)
            gen_data.append(tournament_size)

            data.append(gen_data)
            # --------------Record data to export----------------

        population = mutated_children
        population[-1] = best # ELITISM
        generations += 1

    print(f'Generations: {generations}')
    return best, best_cost, data

def ga_csp_novel(pop_size,crossover_prob, mutation_prob, tournament_size, stocks, stock_price, order_len, order_q, time_limit):
    population = generate_random_population(pop_size, stocks, order_len, order_q)
    population_cost = generate_population_cost(population,stocks, stock_price, order_len, order_q )
    gen_best_sol = min(zip(population,population_cost), key= lambda x: x[1]) # Tuple of participant and cost

    best = gen_best_sol[0]
    best_cost = gen_best_sol[1]

    generations = 1
    start_time = time.time()
    end_time = time.time() + time_limit

    parents_cost = []
    mutated_children_cost = []


    cost_history = []
    average_cost_history = []
    average_cost = sum(population_cost) / len(population_cost)

    mutation_index = 0
    mutation_probs = [1,0.9,0.8,0.6,0.4,0.2,0.1,0]
    crossover_index = 0
    crossover_probs = [0,0.1,0.2,0.4,0.6,0.8,0.9,1]

    tabu = []

    # DATA to EXPORT
    data = []

    while time.time() < end_time:
        elapsed_time = time.time() - start_time
        gen_data = []

        # PARENT SELECTION
        r = random.random()
        if r < 1:
            roulette_result = roulette_selection(population, population_cost, pop_size, 5,stocks, stock_price, order_len, order_q)
        elif r < 0.9: 
            roulette_result = tournament_selection(tournament_size,population, population_cost, pop_size,stocks, stock_price, order_len, order_q)
        else:
            roulette_result = random_selection(population, population_cost, pop_size)

        # Calculate all parents cost here->
        parents = roulette_result[0]
        parents_cost = roulette_result[1]
        # print(f'Parents after parent selection: {len(parents)}')

        # CROSSOVER
        children = []
        parents_copy = parents
        if random.random() < crossover_prob:
            while len(parents_copy) > 1:
                pair = parents_copy[-2:]
                parents_copy = parents_copy[:-2]
                if random.random() < 0.5:
                    s1 = crossover_uniform(pair[0], pair[1])
                    s2 = crossover_uniform(pair[0], pair[1])
                    # s1 = global_recombination(pair[0], population)
                    # s1 = global_recombination(pair[1], population)
                else:
                    s1,s2 = crossover(pair[0], pair[1])
                children.append(s1)
                children.append(s2)
        else: children = parents_copy


        # MUTATION
        mutated_children = []
        for child in children:
            n = random.random()
            if n < mutation_prob:
                if n < 0.9:
                    mutated_child = mutation(child)
                elif n < 0:
                    mutated_child = row_mutation(child)
                elif n < 1:
                    mutated_child = large_mutation(child)
                else:
                    mutated_child = column_mutation(child)
            else: 
                mutated_child = child
            mutated_children.append(mutated_child)

        # Calculate all mutated children cost here->
        mutated_children = np.array(mutated_children)
        mutated_children_cost = generate_population_cost(mutated_children, stocks, stock_price, order_len, order_q )



        # SURVIVOR SELECTION - Combine parents and mutated children, apply selection
        children_and_parents = np.vstack((mutated_children, population))
        children_and_parents_cost = mutated_children_cost + list(population_cost)
        # print(f'Number of children + parents + population: {len(children_and_parents)}') #Bcomment

        #region Replacement SURVIVOR SELECTION
        # population = mutated_children
        # population_cost = mutated_children_cost
        #endregion

        #region Roulette - Tournament - Random SURVIVOR SELECTION
        r = random.random()
        if r < 0.5:
            survivors = roulette_selection(children_and_parents, children_and_parents_cost, pop_size, 6, stocks, stock_price, order_len, order_q)
        elif r < 0.8: 
            survivors = tournament_selection(tournament_size, children_and_parents, children_and_parents_cost, pop_size, stocks, stock_price, order_len, order_q)
        else:
            survivors = random_selection(children_and_parents, children_and_parents_cost, pop_size)
        # Set the survivors as the population
        population = survivors[0]
        population_cost = survivors[1]
        #endregion

        #region BCSO - Back Controlled Selection Operator Test SURVIVOR SELECTION
        # bcso_survivors = []
        # bcso_survivors_cost = []
        # if len(cost_history) >= 60:
        # # if len(cost_history) >= 2 and ((end_time - time.time()) / (end_time- start_time) < 0.8):
        #     while len(bcso_survivors) < pop_size:
        #         selected_index = random.randint(0, len(children_and_parents) - 1)
        #         sol = children_and_parents[selected_index]
        #         sol_cost = children_and_parents_cost[selected_index]
        #         # If the current solution's cost is less than the average of the last generation it will be selected
        #         if sol_cost <= average_cost_history[-5]:
        #             bcso_survivors.append(sol)
        #             bcso_survivors_cost.append(sol_cost)
        #         else: 
        #             continue
        #     population = bcso_survivors
        #     population_cost = bcso_survivors_cost
        # else:
        #     population = mutated_children
        #     population_cost = mutated_children_cost
        #endregion######

        #region Top n SURVIVOR SELECTION
        # sorted_population_with_cost = sorted(zip(children_and_parents, children_and_parents_cost), key= lambda x: x[1])
        # sorted_population, sorted_population_cost = zip(*sorted_population_with_cost)
        # sorted_population = list(sorted_population)
        # sorted_population_cost = list(sorted_population_cost)
        # s = int(pop_size* 2)
        # population = sorted_population[:s]
        # population_cost = sorted_population_cost[:s]
        # population += sorted_population[-pop_size:]
        # population_cost += sorted_population_cost[-pop_size:]
        #endregion

        #region Add random individuals to the population to increase diversity
        # if generations % 20 == 0:
        #     random_population = generate_random_population(int(pop_size * 1),stocks, order_len, order_q)
        #     random_pop_cost = generate_population_cost(random_population, stocks, stock_price, order_len, order_q)
        #     population += list(random_population)
        #     population_cost += list(random_pop_cost)

            # # Periodic reinitialisation if not improving
            # population = generate_random_population(pop_size, stocks, order_len, order_q)
            # population_cost = generate_population_cost(population, stocks, stock_price, order_len, order_q)
            # print(f'Just added new random solutions to the population now of size: {len(population)}') #Bcomment
        #endregion

        #region ############# Tabu Search ########################
        np.append(tabu, population[-10:])
        if len(tabu) > 100:
            tabu = tabu[-100:]
        
        population = np.array(population)
        tabu = np.array(tabu)

        population = [x for x in population if not any(np.array_equal(x, t) for t in tabu)]
        #endregion ############# Tabu Search ########################

        # UNIQUE solutions
        unique_elements = np.unique(population_cost)
        n_unique = len(unique_elements)

        # UPDATE BEST SOLUTION and COST
        gen_best_sol = min(zip(population,population_cost), key= lambda x: x[1]) # Tuple of participant and cost
        if gen_best_sol[1] < best_cost:
            best = gen_best_sol[0]
            best_cost = gen_best_sol[1]
            print(f'new best found at time {round(time.time() - start_time)}: {best_cost} pop-size: {pop_size}, unique:{n_unique}, mutation: {mutation_prob}, cross:{crossover_prob}, len population: {len(population)}')
            
            # --------------Record data to export----------------    
            gen_data.append(generations)
            gen_data.append(elapsed_time)
            gen_data.append(best_cost)
            gen_data.append(average_cost)
            gen_data.append(len(population))
            gen_data.append(mutation_prob)
            gen_data.append(crossover_prob)
            gen_data.append(tournament_size)
            gen_data.append(n_unique)

            data.append(gen_data)
            # --------------Record data to export----------------

        #region ############# Local Search #########################
        # if generations % 100 == 0:
        #     best = local_search(best)
        #     best_cost = evaluate_csp(best, stocks, stock_price, order_len, order_q)

        #endregion ############# Local Search #########################

        # Elitism
        population[-1] = best
        population_cost[-1] = best_cost
        # print(f'Population after survivor selection: {len(population)}')
        average_cost = sum(population_cost) / len(population_cost)
        cost_history.append(best_cost)
        average_cost_history.append(average_cost)

        # Do something everytime the best cost doesn't improve for n generations
        if len(cost_history) > 5 and best_cost == cost_history[-5]:
        #         # mutation_index = (mutation_index + 1) % len(mutation_probs)
        #         # mutation_prob = mutation_probs[mutation_index]
        #         # crossover_index = (crossover_index + 1) % len(crossover_probs)
        #         # crossover_prob = crossover_probs[crossover_index]
        #         # print(f'Picked new mutation prob of {mutation_probs[mutation_index]}')
        #         # print(f'Picked new mutation prob of {crossover_probs[crossover_index]}')
                if generations % 10 == 0:
                    # print(f'Reset pop')
                    population = generate_random_population(pop_size, stocks, order_len, order_q)
                    population_cost = generate_population_cost(population, stocks, stock_price, order_len, order_q)

        #         pop_size += int(pop_size/10)
        #         pop_size = min(pop_size,100)
        # else:
        #         pop_size -= int(pop_size/10)
        #         pop_size = max(pop_size,20)

        generations += 1

    print(f'Solution:\n{best}')
    print(f'Cost:{best_cost}')
    print(f'Generations: {generations}')
    # print(f'Population: {pop_size}')

    return best, best_cost, data

def export_csv(data, header, file_name):
    with open(file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in data:
                writer.writerow(row)

def run_base_n_times(n, time_limit):
    all_data = []
    for _ in range(n):
        best, best_cost, data = ga_csp_base(6, 1, 5, stocks, stock_price, order_len, order_q, time_limit)
        if n == 1:
            all_data = data
        else:
            all_data.append(data[-1])
    return best, best_cost, all_data

def run_novel_n_times(n, time_limit):
    all_data = []
    for _ in range(n):
        best, best_cost, data = ga_csp_novel(6, 0.1, 1, 5, stocks, stock_price, order_len, order_q, time_limit)
        if n == 1:
            all_data = data
        else:
            all_data.append(data[-1])
    return best, best_cost, all_data



#----------------------------------------------MAIN--------------------------------------------------------------

# best, best_cost, data = ga_csp_base(6, 1, 5, stocks, stock_price, order_len, order_q, 10)
# best, best_cost = ga_csp_novel(6, 0.1, 1, stocks, stock_price, order_len, order_q, 60)
# print(best)
# print(best_cost)


time_limit = 10
#------------BASE export-------------------------
base_header = ['generation', 'time', 'best cost', 'population', 'mutation', 'tournament']

# best, best_cost, data = run_base_n_times(1, 1)
# export_csv(data, base_header, 'baseline.csv')

# best, best_cost, data = run_base_n_times(30, 30)
# export_csv(data, base_header, 'baseline_n_runs.csv')


#------------NOVEL export-------------------------
novel_header = ['generation', 'time', 'best cost', 'average cost', 'population', 'mutation', 'crossover', 'tournament', 'unique']

best, best_cost, data = run_novel_n_times(1, 30)
# export_csv(data, novel_header, 'novel.csv')

# best, best_cost, data = run_novel_n_times(30, time_limit=2)
# export_csv(data, novel_header, 'novel_n_runs.csv')





#region Testing

# best, best_cost = random_csp(stocks, stock_price, order_len, order_q, 30)
# test = random_csp(stocks, stock_price, order_len, order_q, 2)
# print(f'Cost: {test[1]}')

# sol1 = random_solution(order_len,order_q,len(stocks))
# print(sol1)
# sol1 = mutation(sol1)
# print()
# print(sol1)




# sol2 = random_solution(order_len,order_q,len(stocks))
# sol3 = crossover_uniform(sol1,sol2)
# print(sol1)
# print(sol2)
# print(sol3)


# sol2 = random_solution(order_len,order_q,len(stocks))
# print('\n',sol2)
# s1, s2 = crossover(sol1, sol2)
# print(f'crossover s1:\n{s1}\n')
# print(f'crossover s2:\n{s2}')

# s1 = np.array([[2, 1, 2],
#                [5, 2, 0],
#                [3, 0, 2]])

# s2 = np.array([[4, 1, 0],
#                [7, 0, 0],
#                [2, 0, 3]])

# cut = random.randint(1,len(order_len)-1)
# print(f'cut {cut}')
# s1_p1 = s1[:cut] 
# s2_p2 = s2[cut:]
# s1s2 = np.vstack((s1_p1,s2_p2))
# s1_p2 = s1[cut:]
# s2_p1 = s2[:cut]
# s2s1 = np.vstack((s2_p1,s1_p2))

# print(f's1 + s2:\n{s1s2}\n')
# print(f's2 + s1:\n{s2s1}')

# Hardcoded solution test
# sol1 = np.array(
# [[ 0,  0,  0,  0,  0,  0,  0,  2],
#  [ 0,  0,  0,  0,  0,  1,  1,  2],
#  [ 1,  1,  1,  0,  0,  0,  0,  1],
#  [ 15,  0,  0, 0,  0,  0,  0,  0],
#  [ 0,  6,  0,  0,  0,  0,  0,  0],
#  [ 0, 11,  0,  0,  0,  0,  0,  0],
#  [ 1,  1,  0,  4,  0,  0,  0,  0],
#  [ 0,  0,  0,  0, 15,  0,  0,  0],
#  [ 0,  1,  0,  0, 12,  0,  0,  0],
#  [ 0,  0,  0,  0,  5,  0,  0,  0],
#  [ 0,  1,  1,  0,  0,  0,  0,  0],
#  [ 0,  0,  7,  2,  0,  0,  0,  0],
#  [ 0,  1,  0,  0,  0,  2,  0,  0],
#  [ 0,  0,  2,  0,  2,  1,  1,  0],
#  [ 0,  0,  1,  0,  6,  0,  0,  3],
#  [ 1,  0,  1,  0,  0,  2,  0,  0],
#  [ 0,  1,  1,  0,  1,  2,  0,  3],
#  [ 1,  0,  0,  1,  0,  0,  0,  1]])
# print(sol1)
# sol1_cost = evaluate_csp(sol1, stocks, stock_price, order_len, order_q)
# print(f'cost: {sol1_cost}')
###################

# population = generate_random_population(10, stocks, order_len, order_q)
# print(population)
# new_population = tournament_selection(2,population, 3,stocks, stock_price, order_len, order_q)
# print('New pop')
# print(new_population)

# mutated = mutation(sol)
# print(mutated)
# make Mutation
# row = np.array([0,0,7,0,2,3,0,0,1])
# print(f'Row before mutation:{row}')
# q = 5
# # Pick an operation to perform on a random element such as +4, no need to include negative elements since the same valued be 
# # subtracted from another element
# # operations = np.arange(1,row.max()+1) 
# # operation = np.random.choice(operations)
# # # print(operations)
# # print(f'Operation to perform: +{operation}')

# indeces = np.arange(0,len(row))
# # indeces = np.arange(3)
# # valid_pick = False
# # while not valid_pick:
# #     i = np.random.choice(indeces)
# #     if row[i] != 0: valid_pick = True

# i = np.random.choice(indeces)
# # print(indeces)
# print(f'index of element getting incremented: {i}')
# # print(f'indeces before removal: {indeces}')
# # Increasing random element by 1
# row[i] += 1

# # Decreasing random element by 1
# indeces = np.where((row > 0) & (indeces != i))[0]
# # indeces = np.delete(indeces, np.where(indeces == i)) # the index whose value is equal to i, simply writing i produces errors since the index might not match with the value
# # print(f'Indeces to choose from to decrease by 1: {indeces}')
# i = np.random.choice(indeces)
# print(f'index of element getting decremented: {i}')
# row[i] -= 1
# print(f'Row after mutation: {row}')

# make crossover

# make parent selection

#####################

# random.seed(1)
# solution = random_solution(order_len,order_q, len(stocks))
# activities = solution_to_activities(solution, stocks,stock_price, order_len, order_q )
# cost = evaluate_csp(activities, stocks, stock_price, order_len)
# print(solution)
# print(activities)
# print(cost)

# dict1 = {50: [30, 20, 20, 20, 20, 20], 80: [30,30,30,25,25,25,25,20,20], 100: [30, 30, 30, 25, 25, 25]}
# # print(dict1)
# dict2 = {}
# for stock in stocks:
#     # print(dict1[stock])
#     orders = dict1[stock]
#     activities = [[]]
#     index = 0
#     while len(orders) > 0:
#         order = orders[index]
#         # If the current order len + last activity's sum is exactly equal to stock then 
#         # it means that I should stop iterating thorugh the orders to check for another order len to add.
#         # It also means that I can add en empty activity so that next iteration I don't have to check that same activity again.
#         if sum(activities[-1]) + order == stock:
#             activities[-1].append(order)
#             orders.remove(order)
#             activity = []
#             activities.append(activity)
#             index = 0
#             continue
#         # I know that current order + sum of last activity is not equal to stock length, so check if it's
#         # less thank stock length, this means that more orders can be added.
#         if sum(activities[-1]) + order < stock:
#             activities[-1].append(order)
#             orders.remove(order)
#             index = 0
#             continue
#         # The current order len doesn't add up to exactly the stock len and goes over the stock length if added to activity
#         else: # sum of last activity + order > stock
#             # Continue checking next order len if there's still orders left that maybe could fit until the end of the list
#             if index != (len(orders)-1):
#                 index += 1
#                 continue # This doens't work because it doesn't add the original order to a new activity but the one on the current iteration. I need to find a way to remember what the original length was 
#             else:
#                 activity = [orders[0]]
#                 orders.remove(order)
#                 activities.append(activity)
#                 index = 0
#                 continue
        
#     dict2[stock] = activities

# print(dict2)
            

# my_dict = {}
# stock_len = [50,80,100]
# order_len = [20,25,30]
# sol = [[1,2,1], [2,4,3], [2,1,1]]

# for i, stock in enumerate(stock_len):
#     my_list = []
#     for j, order in enumerate(order_len):
#         my_list.extend([order] * sol[i][j])
#     my_dict[stock] = my_list
# print(my_dict)

# my_list = []
# my_list.extend([25] *3 + [30] * 4)
# my_list.reverse()
# print(my_list)

# l50 = np.transpose(s1)[0]
# total_stock_order = l50 * order_len
# total_stock_len_l50_price = math.ceil(sum(total_stock_order)/stocks[0])*stock_price[0]
# print(l50)
# print(f'Stock times order len: {total_stock_order}')
# print(f'Total stock price for L-50: {total_stock_len_l50_price}')
#endregion

#region Best Results
# Problem 1: 4460
# Problem 1: 4426
# Problem 1: 4440 time 34 in a run of 360s
# Problem 1: 4456 time 29 in a run of 60s pop = 5, mutation = 1
# Problem 1: 4464 time 19 in a run of 60s pop = 15, mutation = 1 tsize = 3
# Problem 1: 4450 time 19 in a run of 60s pop = 15, mutation = 1 tsize = 2
# Problem 1: 4392 time 59 in a run of 60s pop = 15, mutation = 1 tsize = 2, 1 row mutation, increase pop_size by 1 every generation
# Problem 1: 4399 time 38 in a run of 60s pop = 15, mutation = 1 tsize = 2, 1 row mutation
# Problem 1: 4393 time 39 in a run of 60s pop = 20, mutation = 1, crossover = 0.1 tsize = 2, 1 row mutation
# Problem 1: 4375 time 51 in a run of 60s pop = 20, mutation = 1, crossover = 0.05 tsize = 2, 1 row mutation
# Problem 1: 4372 time 19 in a run of 60s pop = 30, mutation = 1, crossover = 0.05 tsize = 2, 1 row mutation
# Problem 1: 4372 time 09 in a run of 60s pop = 10, mutation = 1, crossover = 0.05 tsize = 2, 1 row mutation, final pop = 26
# Problem 1: 4362 time 42 in a run of 60s pop = 100, mutation = 0.6, crossover = 0.05 tsize = 2, 1-len(sol) row mutation, final pop = 84, evenry 50 gen reduce pop by 2, every 100 gen add 20% new random pop
# Problem 1: 4356 time 43 in a run of 180s pop = 100, mutation = 0.6, crossover = 0.05 tsize = 2, len(sol) -1 row mutation, final pop = 82, evenry 100 gen reduce pop by 2, every 100 gen add 20% new random pop

# AFTER FIXING DECODING FUNCTION (SOLUTION TO ACTIVITIES WHERE I WAS NOT USING FIRST FIT DESCENDING SINCE MY ORDERS WERE IN ASCENDING)
# Problem 1: 4165 time 41 in a run of 60s pop = 100, mutation = 0.6, crossover = 0.1 tsize = 2, len(sol) -1 row mutation, final pop = 10, evenry 100 gen reduce pop by 6, every 100 gen add 20% new random pop
# Problem 1: 4156 time 41 in a run of 60s pop = 100, mutation = 0.6, crossover = 0.1 tsize = 2, len(sol) -1 row mutation, final pop = 10, evenry 100 gen reduce pop by 2, every 100 gen add 20% new random pop
# Problem 1: 4200 time 41 in a run of 60s pop = 100, mutation = 1, crossover = 0.1 tsize = 2, len(sol) -1 row mutation, final pop = 10, evenry 100 gen reduce pop by 2, every 100 gen add 20% new random pop
# Problem 1: 4145 time 44 in a run of 60s pop = 100, mutation = 1, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 88, evenry 100 gen reduce pop by 2, every 100 gen add 20% new random pop, BCSO
# Problem 1: 4133 time 96 in a run of 180s pop = 100, mutation = 1, crossover = 0.05 tsize = 2, len(sol) -1 row mutation, final pop = 24, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop
# Problem 1: 4096 time 161 in a run of 180s pop = 300, mutation = 1, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 280, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection
# Problem 1: 4111 time 51 in a run of 60s pop = 300, mutation = 1, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 294, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection after 60 gens
# Problem 1: 4107 time 60 in a run of 60s pop = 300, mutation = 1, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 294, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection after 60 gens, BCSO survivors < best 5th cost history
# Problem 1: 4169 time 60 in a run of 60s pop = 10, mutation = 1, crossover = 1 tsize = 2, len(sol) -1 row mutation, loop-based slow random initialisation
# Problem 1: new best found at time 14: 4116 pop-size: 20, unique:14, mutation: 1, cross:0.1, len population: 44
# Problem 1: new best found at time 12: 4114 pop-size: 200, unique:101, mutation: 1, cross:0.1, len population: 440
# Problem 1: new best found at time 24: 4082 pop-size: 200, unique:93, mutation: 1, cross:0.1, len population: 420
# Problem 1: new best found at time 58: 4077 pop-size: 5, unique:4, mutation: 1, cross:0.1, len population: 5
# Problem 1: new best found at time 43: 4072 pop-size: 6, unique:5, mutation: 1, cross:0.1, len population: 6
# Problem 1: new best found at time 57: 4049 pop-size: 6, unique:4, mutation: 1, cross:0.1, len population: 6
# Problem 1: new best found at time 47: 4041 pop-size: 6, unique:3, mutation: 1, cross:0.1, len population: 6
# Problem 1: new best found at time 42: 4029 pop-size: 6, unique:7, mutation: 1, cross:0.1, len population: 12
# Problem 1: new best found at time 45: 4022 pop-size: 6, unique:4, mutation: 1, cross:0.1, len population: 12



# Problem 2: 1796.0 time 55 in a run of 60s pop = 30, mutation = 1, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 18, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection after 60 gens, BCSO survivors < best 5th cost history
# Problem 2: 1797.0 time 59 in a run of 60s pop = 30, mutation = 0.4, crossover = 0 tsize = 2, len(sol) -1 row mutation, final pop = 18, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection after 60 gens, BCSO survivors < best 5th cost history
# Problem 2: 1798.0 time 59 in a run of 60s pop = 30, mutation = 0.4, crossover = 0.05 tsize = 2, len(sol) -1 row mutation, final pop = 18, evenry 100 gen reduce pop by 2, every 50 gen add 80% new random pop, BCSO survivor selection after 60 gens, BCSO survivors < best 5th cost history
# Problem 2: 1792.5 time 59 in a run of 60s pop = 20, mutation = 1, crossover = 0 tsize = 4 90% , 10% random parent selection, len(sol) -1 row mutation, top survivor selection
# Problem 2: 1792.0 time 59 in a run of 60s pop = 30, mutation = 1, crossover = 0.05 tsize = 4 90% , 10% random parent selection, len(sol) -1 row mutation, no survivor selection
# Problem 2: 1789.5 time 54 in a run of 60s pop = 30, mutation = 1, crossover = 0.05 tsize = 4 90% , 10% random parent selection, len(sol) -1 row mutation, pop-size top survivor selection
# Problem 2: 1787.5 time 49 in a run of 60s pop = 30, mutation = 1, crossover = 0.05 tsize = 4 90% , 10% random parent selection, len(sol) -1  mutation + column + row mutation 20% 10% 70%, pop-size top survivor selection,

#endregion

