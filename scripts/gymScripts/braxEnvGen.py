import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from carl.context.selection import StaticSelector
from carl.envs import CARLBraxAnt


def generate_AntEnvs(parameter):
    # {'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 'tau': 0.02, 'theta_threshold_radians': 12.0, 'x_threshold': 2.4}

    gravity_list = [-0.98, -1.09, -1.23, -1.4, -1.63, -1.96, -2.45, -3.27, -4.9, -9.8, -19.6]
    
    friction_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    
    elasticity_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    
    ang_damping_list = [-0.1, -0.05, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    mass_torso_list = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0]
    
    viscosity_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]


    if parameter == 'gravity':
        contexts_list = gravity_list
    elif parameter == 'friction':
        contexts_list = friction_list
    elif parameter == 'elasticity':
        contexts_list = elasticity_list
    elif parameter == 'damping':
        contexts_list = ang_damping_list
    elif parameter == 'torso':
        contexts_list = mass_torso_list
    elif parameter == 'viscosity':
        contexts_list = viscosity_list
    else:
        raise ValueError('Invalid parameter')
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLBraxAnt.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLBraxAnt.get_default_context()}
    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLBraxAnt(context_selector=StaticSelector(contexts=contexts))

    return env, contexts


def generate_AcrobotEnvs(parameter):
    # {'LINK_LENGTH_1': 1.0, 'LINK_LENGTH_2': 1.0, 'LINK_MASS_1': 1.0, 'LINK_MASS_2': 1.0, 'LINK_COM_POS_1': 0.5, 'LINK_COM_POS_2': 0.5, 'LINK_MOI': 1.0, 'MAX_VEL_1': 12.5663706144, 'MAX_VEL_2': 28.2743338823, 'torque_noise_max': 0.0, 'INITIAL_ANGLE_LOWER': -0.1, 'INITIAL_ANGLE_UPPER': 0.1, 'INITIAL_VELOCITY_LOWER': -0.1, 'INITIAL_VELOCITY_UPPER': 0.1}

    # generate link_length_1 list, (0.1, 10, <class ‘float’>)
    link_length_1_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # link_length_1_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # link_length_1_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # generate link_length_2 list, (0.1, 10, <class ‘float’>)
    link_length_2_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # generate link_mass_1 list, (0.1, 10, <class ‘float’>)
    link_mass_1_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # generate link_mass_2 list, (0.1, 10, <class ‘float’>)
    link_mass_2_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # generate max_velocity_1 list, (1.2566370614359172, 125.66370614359172, <class ‘float’>)
    max_velocity_1_list = [1.2566370614359172, 2.5132741228718345, 3.769911184307752, 5.026548245743669, 6.283185307179586, 12.566370614359172, 25.132741228717345, 125.66370614359172]
    # generate max_velocity_2 list, (2.827433388231188, 282.7433388231188, <class ‘float’>)
    max_velocity_2_list = [2.827433388231188, 5.654866776462376, 8.482300164693564, 11.309733552924751, 14.137166941155939, 28.274333882311878, 56.548667764623756, 282.743338231188]
    # generate torque_noise_max list, (0.0, 0.1, <class ‘float’>)
    torque_noise_max_list = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]

    if parameter == 'LINK_LENGTH_1':
        contexts_list = link_length_1_list
        context_name = 'LINK_LENGTH_1'
    elif parameter == 'LINK_LENGTH_2':
        contexts_list = link_length_2_list
        context_name = 'LINK_LENGTH_2'
    elif parameter == 'LINK_MASS_1':
        contexts_list = link_mass_1_list
        context_name = 'LINK_MASS_1'
    elif parameter == 'LINK_MASS_2':
        contexts_list = link_mass_2_list
        context_name = 'LINK_MASS_2'
    elif parameter == 'MAX_VEL_1':
        contexts_list = max_velocity_1_list
        context_name = 'MAX_VEL_1'
    elif parameter == 'MAX_VEL_2':
        contexts_list = max_velocity_2_list
        context_name = 'MAX_VEL_2'
    elif parameter == 'torque_noise_max':
        contexts_list = torque_noise_max_list
        context_name = 'torque_noise_max'
    else:
        raise ValueError('Invalid parameter')

    # # generate link_length_1
    # context_length_1_list = []
    # for link_length_1 in link_length_1_list:
    #     context_length_1 = CARLAcrobot.get_default_context().copy()
    #     context_length_1['LINK_LENGTH_1'] = link_length_1
    #     context_length_1_list.append(context_length_1)

    # contexts = {0: CARLAcrobot.get_default_context()}
    # for i, context_length_1 in enumerate(context_length_1_list):
    #     contexts[i+1] = context_length_1

    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLAcrobot.get_default_context().copy()
        env_context[context_name] = c
        env_contexts.append(env_context)

    contexts = {0: CARLAcrobot.get_default_context()}
    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLAcrobot(context_selector=StaticSelector(contexts=contexts))

    return env, contexts

    # for i in range(len(contexts)):
    #     env.context_id = i
    #     # render = lambda: plt.imshow(env.render())
    #     env.reset()
    #     print("Currently using link1 length: ", env.context['LINK_LENGTH_1'])

if __name__ == '__main__':
    pass