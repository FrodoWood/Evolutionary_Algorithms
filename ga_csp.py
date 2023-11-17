import random
import numpy as np
import time
import copy
import math

# stocks = [4300, 4250, 4150, 3950, 3800, 3700, 3550, 3500]
# stock_price = [86, 85, 83, 79, 68, 66, 64, 63]
# order_len = [2350, 2250, 2200, 2100, 2050, 2000, 1950, 1900, 1850, 1700, 1650, 1350, 1300, 1250, 1200, 1150, 1100, 1050]
# order_q = [2, 4, 4, 15, 6, 11, 6, 15, 13, 5, 2, 9, 3, 6, 10, 4, 8, 3]

# stocks = [10,13,15]
# stock_price = [100,130,150]
# order_len = [3,4,5,6,7,8,9,10]
# order_q = [5,2,1,2,4,2,1,3]

stocks = [50,80,100]
stock_price = [100,175,250]
order_len = [20,25,30]
order_q = [5,7,5]



def random_solution(order_lengths, order_q, n_stocks):
    solution = np.zeros(shape=(len(order_lengths), n_stocks))
    for index, order_len in enumerate(order_lengths):
        quantity = order_q[index]
        quantity_left  = quantity
        activity = np.zeros(shape=(n_stocks))
        # print(activity)
        for i in range(n_stocks):
            random_q = random.randint(0,quantity_left)
            if sum(activity) == quantity:
                break
            if(i == n_stocks-1):
                activity[i] = quantity_left
                break
            activity[i] = random_q
            quantity_left -= random_q
        solution[index] = activity
    return solution
            
def evaluate_csp(solution, stocks, stock_price, order_len):
    sol_by_stock_len = np.transpose(solution)
    cost = np.zeros(shape= sol_by_stock_len.shape[0])
    for index, stock_len in enumerate(sol_by_stock_len):
        # Calculate the total length of stock needed for each order length
        total_stock_order = stock_len * order_len
        # Calculate the number of stock lengths needed in total for all order lengths, and then multiply that by the stock price.
        total_stock_len_price = math.ceil(sum(total_stock_order) / stocks[index]) * stock_price[index]
        cost[index] = total_stock_len_price
    return sum(cost)

def solution_to_activities(solution, stocks, stock_price, order_len, order_q):
    sol = np.transpose(solution).astype(int)
    my_dict = {}
    for i, stock in enumerate(stocks):
        my_list = []
        for j, order in enumerate(order_len):
            my_list.extend([order] * sol[i][j])
        my_list.reverse()
        my_dict[stock] = my_list
    print(my_dict)



def random_csp(stocks, stock_price, order_len, order_q, time_limit):
    end_time = time.time() + time_limit
    best_sol = []
    best_sol_cost = float('inf')
    iterations = 0
    while time.time() < end_time:
        sol = random_solution(order_len,order_q, len(stocks))      
        sol_cost = evaluate_csp(sol, stocks, stock_price, order_len)
        # If the cost is lower than best cost then make s1 the best sol
        if(sol_cost < best_sol_cost):
            best_sol = sol
            best_sol_cost = sol_cost
        iterations += 1
    print(f'Iterations : {iterations}')
    return best_sol, best_sol_cost


# test = random_csp(stocks, stock_price, order_len, order_q, 2)
# print(f'Cost: {test[1]}')

# test = random_solution(order_len,order_q,len(stocks))
# print(test)




#region Testing
# random.seed(42)
# solution = random_solution(order_len,order_q, len(stocks))
# activity_test = solution_to_activities(solution, stocks,stock_price, order_len, order_q )

dict1 = {50: [30, 20, 20, 20, 20, 20], 80: [30, 25, 25, 25, 25], 100: [30, 30, 30, 25, 25, 25]}
print(dict1)
dict2 = {}
for stock in stocks:
    # print(dict1[stock])
    orders = dict1[stock]
    activities = [[]]
    for order in orders:
        if sum(activities[-1]) + order <= stock:
            activities[-1].append(order)
        else:
            activity = [order]
            activities.append(activity)
    # print(activities)
    dict2[stock] = activities
print(dict2)
            



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
