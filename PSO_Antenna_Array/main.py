from antennaarray import AntennaArray
from particle import Particle
import random
import numpy as np
import math
import time
# random.seed(42)

# A simple example of how the AntennaArray class could be used as part
# of a random search.

def random_parameters(antenna_array_problem):
    b = antenna_array_problem.bounds()  
    # params = [low + random.random()*(high-low) for [high, low] in b]
    # print(params)
    return [low + random.random()*(high-low) for [high, low] in b]

# Construct an instance of the antenna array problem with 3 antennae and a
# steering angle of 45 degree.
# antenna_array_problem = AntennaArray(3,90)

###############################################################################
# NOTE: This attempt at solving the problem will work really badly! We        #
# haven't taken constraints into account when generating random parameters.   #
# The probability of randomly generating a design which meets the aperture    #
# size constraint is close to zero. This is just intended as an illustration. #
###############################################################################

# Generate N_TRIES random parameters and measure their peak SLL on the problem,
# saving the best parameters.
# N_TRIES = 100
# best_parameters = random_parameters(antenna_array_problem)
# best_sll = antenna_array_problem.evaluate(best_parameters)
# for _ in range(N_TRIES - 1):
#     parameters = random_parameters(antenna_array_problem)
#     # Note: in this example we are not testing parameters for validity. The
#     # evaluate function penalises invalid solutions by assigning them the
#     # maximum possible floating point value.
#     sll = antenna_array_problem.evaluate(parameters)
#     if sll < best_sll:
#         best_sll = sll
#         best_parameters = parameters

# print("Best peak SLL after {} iterations based on random initialisation: {}".format(
#   N_TRIES, best_sll))
# print(f'Best parameters: {best_parameters}')
  
###############################################################################
# How can we improve on the above attempt? By trying to generate initial      #
# parameters which meet the aperture size constraint!                         #
###############################################################################

def constrained_random_parameters(antenna_array_problem):
    b = antenna_array_problem.bounds()  
    design = [low + random.random()*(high-low) for [high, low] in b]
    design[-1] = antenna_array_problem.n_antennae/2 #Setting the aperture size -> We want a fixed aperture size equal to half the number of antennae
    return design
    
#region ################## Contrained random parameters ######### Pre-written code #############
## Try random search again with this new method of generating parameters ##############
# N_TRIES = 10
# best_parameters = constrained_random_parameters(antenna_array_problem)
# best_sll = antenna_array_problem.evaluate(best_parameters)
# for _ in range(N_TRIES - 1):
#     parameters = constrained_random_parameters(antenna_array_problem)
#     sll = antenna_array_problem.evaluate(parameters)
#     print(sll)
#     if sll < best_sll:
#         best_sll = sll
#         best_parameters = parameters

# print("Best peak SLL after {} iterations based on random initialisation: {}".format(
#   N_TRIES, best_sll))
#endregion ################# Contrained random parameters ######### Pre-written code #############

##################### LAB 4 ############################################################
def valid_constrained_random_parameters(antenna_array_problem):
    isValid = False
    while not isValid:
        b = antenna_array_problem.bounds()  
        design = [low + random.random()*(high-low) for [high, low] in b]
        design[-1] = antenna_array_problem.n_antennae/2 #Setting the aperture size -> We want a fixed aperture size equal to half the number of antennae
        isValid = antenna_array_problem.is_valid(design)
        # if(not isValid):
        #     print("invalid random parameters!")
    return design

def valid_constrained_random_parameters_2(antenna_array_problem, bounds):
    design = []
    n = antenna_array_problem.n_antennae
    bounds = generate_bounds(antenna_array_problem)
    for low,high in bounds:
        design.append(random.uniform(low,high))
    # design.append(n/2)
    return design

def generate_bounds(antenna_array_problem):
    b = antenna_array_problem.bounds()  
    n = antenna_array_problem.n_antennae
    # offset = np.random.uniform(-0.25,0.25)
    prob = random.random()
    offset = -0.25 if prob <= 0.5 else 0.25
    # print(offset)
    unit = (n/2 - (n-1) * 0.25) / (n-1)
    pairs = [[0,unit + offset]]
    for index in range(n-2):
        pair =[]
        pair.append(pairs[index][1]+0.25)
        pair.append(pair[0]+unit)
        pairs.append(pair)
    pairs.append([n/2,n/2])    
    pairs = [[round(x,4) for x in pair] for pair in pairs]
    pairs[0][0] = 0
    pairs[-2][1] = (n/2) - 0.25
    return np.array(pairs)

def generate_valid_bounded_particles(antenna_array_problem, particles_amount, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF):
    particles = []
    for _ in range(particles_amount):
        bounds = generate_bounds(antenna_array_problem)
        initial_pos_X = np.array(valid_constrained_random_parameters_2(antenna_array_problem, bounds))
        initial_vel = np.array(valid_constrained_random_parameters_2(antenna_array_problem, bounds))

        particle = Particle(initial_pos_X, initial_vel, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF, antenna_array_problem.evaluate(initial_pos_X),bounds)
        particles.append(particle)
        # print(particle.pos_X)
    return particles

def generate_valid_particles(antenna_array_problem, particles_amount, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF):
    particles = []
    for _ in range(round(particles_amount*3)):
        initial_pos_X = np.array(valid_constrained_random_parameters(antenna_array_problem))
        initial_pos_X = np.array(sorted(initial_pos_X))
        initial_vel = np.array(valid_constrained_random_parameters(antenna_array_problem))
        particle = Particle(initial_pos_X, initial_vel, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF, antenna_array_problem.evaluate(initial_pos_X),[])
        particles.append(particle)

    # bounded_particles = generate_valid_bounded_particles(antenna_array_problem,particles_amount,INERTIA,COGNITIVE_COEFF,SOCIAL_COEFF)
    # particles.extend(bounded_particles)
    particles.sort(key= lambda particle: particle.p_best_eval)
    particles = particles[:particles_amount]
    return particles

def linear_interpolation(start, end,time_limit,elapsed_time):
    line = (end - start) / time_limit
    return start + (line*elapsed_time)

def pso_always_valid(particles, time_limit):
    g_best = []
    g_best_eval = float('inf')
    g_best_eval_history = []
    result = []

    #Setting up timer
    start_time = time.time()
    end_time = start_time + time_limit

    # Update global best once before loop
    for particle in particles:
        # particle.p_best_eval = antenna_array_problem.evaluate(particle.p_best)
        if(particle.p_best_eval < g_best_eval):
            g_best = particle.p_best
            g_best_eval = antenna_array_problem.evaluate(g_best)
            # print(f'New Global best at time {time.time() - start_time}: {g_best} with eval: {g_best_eval}')

    ## Time limit based algorithm
    while time.time() < end_time:
        # Update each particle
        for particle in particles:
            #region If the difference of eval between the 2 evals is very low, we might reached a local optimum, so reset the population
            # if len(g_best_eval_history) > 1 and (g_best_eval_history[-1] - g_best_eval_history[-2]) < -0.2:
            #     particles = []
            #     g_best_eval_history = []
            #     for _ in range(particles_amount):
            #         initial_pos_X = np.array(valid_constrained_random_parameters(antenna_array_problem))
            #         initial_vel = np.array(valid_constrained_random_parameters(antenna_array_problem))
            #         particle = Particle(initial_pos_X, initial_vel, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF, antenna_array_problem.evaluate(initial_pos_X))
            #         particles.append(particle)
            #endregion
            r1 = np.random.random(antenna_array_problem.n_antennae)
            r2 = np.random.random(antenna_array_problem.n_antennae)

            new_vel = particle.update_velocity(INERTIA - 0.2 ,COGNITIVE_COEFF,r1,1.2*SOCIAL_COEFF,r2, g_best)
            new_pos_X = particle.update_position(new_vel)
            # print(f'these my bounds: {particle.bounds}')
            new_pos_X_clipped = np.clip(new_pos_X,particle.bounds[:, 0],particle.bounds[:, 1])

            # if(antenna_array_problem.is_valid(new_pos_X_clipped) == False): 
            #     print(f'Updating pos resulted in invalid design.xxxxxxxxxx')
            #     # improvement_ratio = 1
            #     # if len(g_best_eval_history) > 1:
            #     #     improvement_ratio = g_best_eval_history[-2] / g_best_eval_history[-1]
            #         # print(improvement_ratio)
            #     # new_vel = particle.update_velocity(INERTIA*0.97, COGNITIVE_COEFF, r1, SOCIAL_COEFF, r2, g_best)
            #     # new_pos_X = particle.update_position(new_vel)
            #     # if(antenna_array_problem.is_valid(new_pos_X) == False):
            #     #     # print('still invalid XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX') 
            #     #     new_vel = particle.update_velocity(INERTIA * random.uniform(1,1.05), 0*COGNITIVE_COEFF, r1, 1.5*SOCIAL_COEFF, r2, g_best)
            #     #     new_pos_X = particle.update_position(-new_vel)

            particle.pos_X = new_pos_X_clipped
            particle.vel = new_vel
            if random.random() >= 0.5:
                particle.bounds = generate_bounds(antenna_array_problem)
            
            # UPDATE PERSONAL BEST
            pos_eval = antenna_array_problem.evaluate(particle.pos_X)
            if(pos_eval < particle.p_best_eval):
                particle.p_best = particle.pos_X
                particle.p_best_eval = antenna_array_problem.evaluate(particle.p_best)
                # UPDATE GLOBAL BEST if a new personal best is found. This saves us from checking global best for every low quality new pos.
                if(particle.p_best_eval < g_best_eval):
                    g_best = particle.p_best
                    g_best_eval = antenna_array_problem.evaluate(g_best)
                    g_best_eval_history.append(g_best_eval)
                    print(f'New Global best at time {time.time() - start_time}: {g_best} with eval: {g_best_eval}')
    
    result.append(g_best)
    result.append(g_best_eval)
    return result
            # print(particle.pos_X, 'personal best ->', particle.p_best, 'eval ->', p_best_eval)

def pso(particles, time_limit):
    g_best = []
    g_best_eval = float('inf')
    result = []
    iteration = 0

    # Update global best once before loop
    for index, particle in enumerate(particles):
        # particle.p_best_eval = antenna_array_problem.evaluate(particle.p_best)
        if(particle.p_best_eval < g_best_eval):
            g_best = particle.p_best
            g_best_eval = antenna_array_problem.evaluate(g_best)
            best_particle = index
            # print(f'New Global best at time {time.time() - start_time}: {g_best} with eval: {g_best_eval}')
    #Setting up timer
    start_time = time.time()
    end_time = start_time + time_limit
    
    ## Time limit based algorithm
    while time.time() < end_time:
        # Update each particle
        for index, particle in enumerate(particles):
            
            r1 = np.random.random(antenna_array_problem.n_antennae)
            r2 = np.random.random(antenna_array_problem.n_antennae)

            new_vel = particle.update_velocity(particle.INERTIA,particle.COGNITIVE_COEFF,r1,particle.SOCIAL_COEFF,r2, g_best)
            new_pos_X = particle.update_position(new_vel)
            particle.pos_X = new_pos_X
            particle.vel = new_vel

            # random_n = random.random()
            # slowdown_prob = 0
            # if((random_n < slowdown_prob) and (antenna_array_problem.is_valid(new_pos_X) == False)):
            #     new_vel = particle.update_velocity(particle.INERTIA, particle.COGNITIVE_COEFF,r1, particle.SOCIAL_COEFF,r2, g_best)
            #     new_pos_X = particle.update_position(new_vel)
            # else:
            #     particle.pos_X = new_pos_X
            #     particle.vel = new_vel

            t = time.time() - start_time
            # Linearly decreasing inertia weight introduced by Shi and Eberhart as described in the paper Particle Swarm Optimization: Basic Concepts, Variants and Applications in Power SystemsYamille
            particle.set_inertia(linear_interpolation(0.9,0.5,time_limit,t))
            # Non-linear inertia weight decrease
            # if(t != 0):
            #     particle.set_inertia(0.9 + (0.3*(t/time_limit)**2) - 0.4*((2*t)/time_limit) + ((math.sin(t/0.6) / 6) * 1/t )  - 0.3 )
            
            # Time varying strategy for adjustment of c1 and c2: IPSO
            new_cog = 2 + (2*(t/time_limit)**2) - 2*((2*t)/time_limit)
            new_soc = 0 - (2*(t/time_limit)**2) + 2*((2*t)/time_limit)
            particle.set_cognitive_coeff(new_cog)
            particle.set_social_coeff(new_soc)
            
            # UPDATE PERSONAL BEST
            pos_eval = antenna_array_problem.evaluate(particle.pos_X)
            if(pos_eval < particle.p_best_eval):
                particle.p_best = particle.pos_X
                particle.p_best_eval = antenna_array_problem.evaluate(particle.p_best)

                # UPDATE GLOBAL BEST if a new personal best is found. This saves us from checking global best for every low quality new pos.
                if(particle.p_best_eval < g_best_eval):
                    g_best = particle.p_best
                    g_best_eval = antenna_array_problem.evaluate(g_best)
                    print(f'New Global best at time {time.time() - start_time}: {g_best} eval->>>> {g_best_eval}')
        # After each iteration
        iteration += 1
    print(f'\nIterations: {iteration}\n')
    result.append(g_best)
    result.append(g_best_eval)
    return result

#region################# Task 1 ##############################################################
# antenna_array_problem = AntennaArray(3,90)
# test_params = [0.5, 1, 1.5]
# test_best_params = [0.848, 0.262, 1.5]
# sll = round( antenna_array_problem.evaluate(test_best_params), 2)
# print(f'Parameters: {test_best_params}, sll: {sll}')
# Parameters: [0.848, 0.262, 1.5], sll: -11.97
#endregion
#region ################## Task 2 ##############################################################
# # Try random search again with this new method of generating parameters ##############
# N_TRIES = 100
# best_parameters = valid_constrained_random_parameters(antenna_array_problem)
# best_sll = antenna_array_problem.evaluate(best_parameters)
# for _ in range(N_TRIES - 1):
#     parameters = valid_constrained_random_parameters(antenna_array_problem)
#     sll = antenna_array_problem.evaluate(parameters)
#     if sll < best_sll:
#         best_sll = sll
#         # print(best_sll)
#         best_parameters = parameters
# print("Best peak SLL after {} valid iterations based on random initialisation: {}".format(
#   N_TRIES, best_sll))

# def update_velocity(inertia, cognitive_coeff, r1, social_coeff, r2):
#     new_vel = inertia * vel + cognitive_coeff * r1 * (p_best - pos_X) + social_coeff * r2 * (g_best - pos_X)
#     return new_vel    

# def update_position(pos_X, vel):
#     return pos_X + vel
#endregion
#region ################## Task 3 ##############################################################
# pos_X = np.array(valid_constrained_random_parameters(antenna_array_problem))
# vel = np.array(valid_constrained_random_parameters(antenna_array_problem))
# vel = (pos_X - vel) / 2
# p_best = pos_X
# g_best = p_best
# INERTIA = 0.721
# COGNITIVE_COEFF = 1.1193
# SOCIAL_COEFF = 1.1193
# population = generate_population(10)
# print(np.array(population))
#endregion
#region Task 3 single particle test
# ################## Task 3 single particle test ##############################################################
# print(f'Start -> {pos_X} with evaluation -> {antenna_array_problem.evaluate(pos_X)}')
# n_iterations = 10
# for i in range(n_iterations):
#     r1 = np.random.random(3)
#     r2 = np.random.random(3)
#     vel = update_velocity(INERTIA,COGNITIVE_COEFF,r1,SOCIAL_COEFF,r2)
#     pos_X = update_position(pos_X,vel)
#     pos_eval = antenna_array_problem.evaluate(pos_X)
#     if(pos_eval < antenna_array_problem.evaluate(p_best)):
#         p_best = pos_X
#     print(pos_X, 'personal best ->', p_best)
#     if(antenna_array_problem.evaluate(p_best) < antenna_array_problem.evaluate(g_best)):
#         g_best = p_best
# print(f'The best -> {g_best} with evaluation -> {antenna_array_problem.evaluate(g_best)}')
# ################## Task 3 single particle test ##############################################################
#endregion

#region Task 3 multiple particle objects test
# ################## Task 3 multiple particle objects test ##############################################################
antenna_array_problem = AntennaArray(4,15)
# INERTIA = 0.721
INERTIA = 0.9
COGNITIVE_COEFF = 1.1193
SOCIAL_COEFF = 1.1193
# particles_amount = round(antenna_array_problem.n_antennae + math.sqrt(antenna_array_problem.n_antennae))
particles_amount = 10
time_limit = 20

print(f'Particles amount: {particles_amount}\n')

#region##################################################### Debugging #######################
# print(antenna_array_problem.bounds())
# design = constrained_random_parameters(antenna_array_problem)
# design_validity = antenna_array_problem.is_valid(design)
# print(f'The design: {design} -> is valid: {design_validity}')
# print(constrained_random_parameters(antenna_array_problem))
# print(antenna_array_problem.is_valid(constrained_random_parameters(antenna_array_problem)))
#endregion##################################################### Debugging #######################

# PSO bounded-----------
# particles_bounded = generate_valid_bounded_particles(antenna_array_problem, particles_amount, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF)
# pso_always_valid_result = pso_always_valid(particles_bounded,time_limit)
# PSO-------------------
particles = generate_valid_particles(antenna_array_problem, particles_amount, INERTIA, COGNITIVE_COEFF, SOCIAL_COEFF)
for index, particle in enumerate(particles):
    print(f'Particle {index} p_best: {particle.p_best_eval}')
print('')
pso_result = pso(particles,time_limit)

# for index, particle in enumerate(particles):
#     print(f'Particle {index} p_best: {round(particle.p_best_eval,6)}    pos_X eval: {round(antenna_array_problem.evaluate(particle.pos_X),6)}')

# print(f'\nGlobal Best: {pso_always_valid_result[0]} eval: {pso_always_valid_result[1]}\n')
print(f'\nGlobal Best: {pso_result[0]} eval: {pso_result[1]}\n')

#region Test results
# best results:
# Problem (3,90):    
# 20 p, 100 iter, inertia = 0.721, (3,90), seed()   Global Best: [0.83993056 0.25583638 1.5] eval: -11.752226543640651
# 20 p,  10 iter, inertia = 0.234, (3,90), seed()   Global Best: [0.84639344 0.26023237 1.5] eval: -12.101438563931527
# 20 p,  58 secs, inertia = 0.721, (3,90), seed()   Global Best: [0.84696459 0.2606647  1.5] eval: -12.122320841525521

# 20 p, 10 iter, inertia = 0.721, (3,90), seed(42) Global Best: [0.85357932 0.26791512 1.5] eval: -11.218617490893799
# 20 p, 10 iter, inertia = 0.234, (3,90), seed(42) Global Best: [0.26041669 0.84660877 1.5] eval: -12.105009266361861
# 20 p, 20 iter, inertia = 0.721, (3,90), seed(42) Global Best: [0.26064845 0.84694262 1.5] eval: -12.121515849792697

# Problem (10,90):
# 20 p, 10 iter, inertia = 0.721, (10,90), seed(42) Global Best: [3.17907791 1.94121622 1.53295847 4.5866988  0.16244243 0.8492348 3.84604734 2.45065786 2.92825978 5.]  eval: 17.834383053880018 
# 20 p, 10 iter, inertia = 0.234, (10,90), seed(42) Global Best: [3.26525184 1.96259124 1.33965387 4.63457329 0.15172089 0.6996159 3.87819971 2.49642676 2.9180364  5.]  eval: 16.112759821273013 
# 5p  , 15 secs, inertia = 0.721, (10,90), seed()   Global Best: [3.43342808 1.59389983 1.10830099 0.13197734 1.90857634 2.78925383 4.25459407 2.36698796 0.67644077 5.] eval: 3.0078877389193415
# 
# Global Best: [9.05169584e-01 3.65336376e+00 2.25150843e+00 2.08496308e-04 2.91851042e+00 4.29074151e+00 5.96799956e-01 1.41829236e+00 1.79046590e+00 5.00000000e+00]
# eval: 7.099732742202844 -> Settings:
# base, 5 p, 20 secs, seed()
# new_vel = particle.update_velocity(INERTIA*0.01,COGNITIVE_COEFF,r1*0,SOCIAL_COEFF*4,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, -new_vel)

# Global Best: [3.39121709 1.10281449 0.46939643 4.34688592 1.95925572 3.87787469 2.76126633 2.41271429 0.08838193 5.]
# eval: 13.877364533906693
# base, 10 p, 20 secs, seed()
# new_vel = particle.update_velocity(INERTIA*0.01,COGNITIVE_COEFF*0,r1,SOCIAL_COEFF*4,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, -new_vel)

#Global Best: [0.59124172 2.09548844 2.46474551 1.17148374 3.82558337 3.15682431 4.36778982 1.53015545 0.32962851 5.] 
# eval: 4.522329776488575
# base, 10p, 10 secs, seed()
# new_vel = particle.update_velocity(INERTIA*0.5,0*COGNITIVE_COEFF,r1,5* SOCIAL_COEFF,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, new_vel)

#Global Best: [1.99367487 1.64692146 1.1107871  2.793966   0.22780737 2.47166502 0.76541166 3.5471456  4.31453889 5.        ] 
# eval: 3.8471961583260694
# new_vel = particle.update_velocity(INERTIA*0.4 ,0*COGNITIVE_COEFF,r1,5* SOCIAL_COEFF,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, new_vel)

#Global Best: [2.43076236 0.6402484  2.9160615  1.48827767 0.25947459 4.28702666 1.9654246  3.61885811 1.11814211 5.] 
# eval: 0.5330286078675226
# new_vel = particle.update_velocity(INERTIA*0.4 ,0*COGNITIVE_COEFF,r1,5* SOCIAL_COEFF,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, new_vel)

#Global Best: [3.16192476 0.21173497 4.21304483 1.85533363 2.48846873 1.55981595 3.56720339 1.03741846 0.75099897 5.] 
# eval: 2.9844065175429266
# base, 10p, 20secs, seed()
# new_vel = particle.update_velocity(INERTIA * 1.05,0*COGNITIVE_COEFF,r1,1.5* SOCIAL_COEFF,r2, g_best)
# new_pos_X = particle.update_position(particle.pos_X, new_vel)

#Global Best: [0.54536682 2.35107047 1.40506505 1.84894335 4.25750056 2.86695263 0.22297732 3.50231401 1.03523514 5.        ] 
# eval: -3.9614128641756627
#base, 10p, 20secs, seed()
#new_vel = particle.update_velocity(INERTIA * 1.05,0*COGNITIVE_COEFF,r1,1.5* SOCIAL_COEFF,r2, g_best)
#new_pos_X = particle.update_position(particle.pos_X, new_vel)

#Global Best: [0.1753358  0.59911103 1.03609606 1.47073773 1.87463257 2.40460565 3.5467454  2.90137337 4.28008357 5.] 
# eval: -3.985371183655716
# inertia 0.729, cog 2, soc 2, 40 seconds
# constriction factor, Chen and Li stochastic approximation theory, and only one pass of new vel and new pos_X

#Global Best: [0.18436195 0.63906607 1.00489146 1.43035308 1.89117004 2.38847556 2.87597544 3.52726519 4.25699167 5.        ] 
# eval: -4.770370115779507
# inertia 0.9, cog 2, soc 2, 20 seconds
# Shi and Eberhart linear inertia decrease
#endregion

#endregion






