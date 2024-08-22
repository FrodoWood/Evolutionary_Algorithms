import numpy as np
import random
from antennaarray import AntennaArray

class Particle:


    def __init__(self, initial_pos_X, initial_velocity, inertia, cognitive_coeff, social_coeff, p_best_eval, bounds):
        self.INERTIA = inertia
        self.COGNITIVE_COEFF = cognitive_coeff
        self.SOCIAL_COEFF = social_coeff
        self.pos_X = initial_pos_X
        self.vel = (initial_pos_X - initial_velocity)
        self.p_best = self.pos_X
        self.p_best_eval = p_best_eval
        self.bounds = bounds


    def update_velocity(self, inertia, cognitive_coeff, r1, social_coeff, r2, g_best):
        return inertia * self.vel + cognitive_coeff * r1 * (self.p_best - self.pos_X) + social_coeff * r2 * (g_best - self.pos_X)

    def update_position(self, vel):
        new_pos_X = self.pos_X + vel
        return new_pos_X 
    
    def set_inertia(self, value):
        self.INERTIA = value

    def set_cognitive_coeff(self, value):
        self.COGNITIVE_COEFF = value

    def set_social_coeff(self, value):
        self.SOCIAL_COEFF_COEFF = value

    def decrease_inertia(self, amount):
        self.INERTIA *= amount
        # self.INERTIA -= random.uniform(0.01,0.03)
        if self.INERTIA < 0.2: self.INERTIA = 0.2
        print(self.INERTIA)
        
    def increase_inertia(self):
        self.INERTIA *= 1.04
        if self.INERTIA > 0.9: self.INERTIA = 0.9
        print(self.INERTIA)

    def randomize_social_coeff(self, low, high):
        self.SOCIAL_COEFF = random.uniform(low, high)
        
    def randomize_cognitive_coeff(self, low, high):
        self.COGNITIVE_COEFF = random.uniform(low, high)
