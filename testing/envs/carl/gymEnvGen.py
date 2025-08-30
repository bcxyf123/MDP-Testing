import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from carl.context.selection import StaticSelector
from carl.envs import CARLAcrobot, CARLCartPole, CARLMountainCar, CARLBipedalWalker, CARLLunarLander, CARLVehicleRacing



def generate_CartPoleEnvs(parameter):
    # {'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0, 'tau': 0.02, 'theta_threshold_radians': 12.0, 'x_threshold': 2.4}

    gravity_list = [0.98, 1.09, 1.23, 1.4, 1.63, 1.96, 2.45, 3.27, 4.9, 9.8, 19.6]
    # generate masscart list, (0.1, 10, <class ‘float’>)
    masscart_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    # generate masspole list, (0.01, 1, <class ‘float’>)
    masspole_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
    # generate length list, (0.1, 5, <class ‘float’>)
    length_list = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2.5, 5.0]
    # generate force_mag list, (1, 100, <class ‘float’>)
    force_mag_list = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    # generate tau list, (0.001, 0.1, <class ‘float’>)
    tau_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    # # generate theta_threshold_radians list, (0.1, 1, <class ‘float’>)
    # theta_threshold_radians_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    # # generate x_threshold list, (0.1, 2.4, <class ‘float’>)
    # x_threshold_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.4]

    if parameter == 'gravity':
        contexts_list = gravity_list
        # default_id = gravity_list.index(9.8)
    elif parameter == 'masscart':
        contexts_list = masscart_list
        # default_id = masscart_list.index(1.0)
    elif parameter == 'masspole':
        contexts_list = masspole_list
        # default_id = masspole_list.index(0.1)
    elif parameter == 'length':
        contexts_list = length_list
        # default_id = length_list.index(0.5)
    elif parameter == 'force_mag':
        contexts_list = force_mag_list
        # default_id = force_mag_list.index(10.0)
    elif parameter == 'tau':
        contexts_list = tau_list
        # default_id = tau_list.index(0.02)
    else:
        raise ValueError('Invalid parameter')
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLCartPole.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLCartPole.get_default_context()}
    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLCartPole(context_selector=StaticSelector(contexts=contexts))

    return env, contexts


def generate_AcrobotEnvs(parameter='LINK_LENGTH_1'):
    # {'LINK_LENGTH_1': 1.0, 'LINK_LENGTH_2': 1.0, 'LINK_MASS_1': 1.0, 'LINK_MASS_2': 1.0, 'LINK_COM_POS_1': 0.5, 'LINK_COM_POS_2': 0.5, 'LINK_MOI': 1.0, 'MAX_VEL_1': 12.5663706144, 'MAX_VEL_2': 28.2743338823, 'torque_noise_max': 0.0, 'INITIAL_ANGLE_LOWER': -0.1, 'INITIAL_ANGLE_UPPER': 0.1, 'INITIAL_VELOCITY_LOWER': -0.1, 'INITIAL_VELOCITY_UPPER': 0.1}

    # generate link_length_1 list, (0.1, 10, <class ‘float’>)
    # link_length_1_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    link_length_1_list = np.arange(0.1, 10.1, 0.05)
    # generate link_length_2 list, (0.1, 10, <class ‘float’>)
    # link_length_2_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    link_length_2_list = np.arange(0.1, 10.1, 0.05)
    # generate link_mass_1 list, (0.1, 10, <class ‘float’>)
    # link_mass_1_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    link_mass_1_list = np.arange(0.1, 10.1, 0.05)
    # generate link_mass_2 list, (0.1, 10, <class ‘float’>)
    # link_mass_2_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
    link_mass_2_list = np.arange(0.1, 10.1, 0.05)
    # generate max_velocity_1 list, (1.2566370614359172, 125.66370614359172, <class ‘float’>)
    max_velocity_1_list = [1.2566370614359172, 1.5, 1.6, 1.7, 1.8, 1.9, 
                           2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 
                           3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 
                           4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 
                           5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
                           6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 
                           7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 
                           8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 
                           9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 
                           10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 
                           11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 
                           12.0, 12.1, 12.2, 12.3, 12.4, 12.566370614359172, 
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                           25.132741228717345, 125.66370614359172]
    # generate max_velocity_2 list, (2.827433388231188, 282.7433388231188, <class ‘float’>)
    max_velocity_2_list = [2.827433388231188, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 
                           21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
                           28.274333882308138, 56.548667764623756, 282.743338231188]
    # generate torque_noise_max list, (0.0, 0.1, <class ‘float’>)
    torque_noise_max_list = np.arange(0.0, 0.1, 0.01)

    if parameter == 'LINK_LENGTH_1':
        contexts_list = link_length_1_list
        context_name = 'LINK_LENGTH_1'
        # default_id = np.where(link_length_1_list == 1.0)[0]
    elif parameter == 'LINK_LENGTH_2':
        contexts_list = link_length_2_list
        context_name = 'LINK_LENGTH_2'
        # default_id = np.where(link_length_2_list == 1.0)[0]
    elif parameter == 'LINK_MASS_1':
        contexts_list = link_mass_1_list
        context_name = 'LINK_MASS_1'
        # default_id = np.where(link_mass_1_list == 1.0)[0]
    elif parameter == 'LINK_MASS_2':
        contexts_list = link_mass_2_list
        context_name = 'LINK_MASS_2'
        # default_id = np.where(link_mass_2_list == 1.0)[0]
    elif parameter == 'MAX_VEL_1':
        contexts_list = max_velocity_1_list
        context_name = 'MAX_VEL_1'
        # default_id = max_velocity_1_list.index(12.566370614359172)
    elif parameter == 'MAX_VEL_2':
        contexts_list = max_velocity_2_list
        context_name = 'MAX_VEL_2'
        # default_id = max_velocity_2_list.index(28.274333882308138)
    elif parameter == 'torque_noise_max':
        contexts_list = torque_noise_max_list
        context_name = 'torque_noise_max'
        # default_id = np.where(torque_noise_max_list == 0.0)[0]
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


def generate_mountaincarEnvs(parameter):

    # {'min_position': -1.2, 'max_position': 0.6, 'max_speed': 0.07, 'goal_position': 0.5, 'goal_velocity': 0, 'force': 0.001, 'gravity': 0.0025, 'start_position': -0.5, 'start_position_std': 0.1, 'start_velocity': 0, 'start_velocity_std': 0.0}

    # generate min_position list, (-1.2, 0.6, <class ‘float’>)
    min_position_list = np.arange(-1.2, 1.2, 0.1)
    # generate max_position list, (-1.2, 0.6, <class ‘float’>)
    max_position_list = np.arange(-0.6, 0.6, 0.1)
    # generate max_speed list, (0, 0.07, <class ‘float’>)
    max_speed_list = np.arange(0.07, 1, 0.01)
    # generate goal_position list, (-1.2, 0.6, <class ‘float’>)
    goal_position_list = np.arange(0.5, 1.0, 0.1)
    # generate goal_velocity list, (0, 0.07, <class ‘float’>)
    goal_velocity_list = np.arange(0.0, 1.0, 0.1)
    # generate force list, (0, 0.07, <class ‘float’>)
    force_list = np.arange(0.001, 0.01, 0.001)
    # generate gravity list, (0, 0.07, <class ‘float’>)
    gravity_list = np.arange(0.0025, 0.01, 0.001)
    # generate start_position list, (-1.5, 0.5, <class ‘float’>)
    start_position_list = np.arange(-0.5, 0.5, 0.1)
    # generate start_position_std list, (0.1, inf, <class ‘float’>)
    start_position_std_list = np.arange(0.1, 1.0, 0.1)
    # generate start_velocity list, (-1.5, 0.5, <class ‘float’>)
    start_velocity_list = np.arange(0.0, 1.0, 0.1)
    # generate start_velocity_std list, (0.1, inf, <class ‘float’>)
    start_velocity_std_list = np.arange(0.0, 1.0, 0.1)

    # default_id = None

    if parameter == 'min_position':
        contexts_list = min_position_list
        # # default_id = min_position_list.index(-1.2)
    elif parameter == 'max_position':
        contexts_list = max_position_list
        # # default_id = max_position_list.index(0.6)
    elif parameter == 'max_speed':
        contexts_list = max_speed_list
        # # default_id = max_speed_list.index(0.07)
    elif parameter == 'goal_position':
        contexts_list = goal_position_list
        # # default_id = goal_position_list.index(0.5)
    elif parameter == 'goal_velocity':
        contexts_list = goal_velocity_list
        # # default_id = goal_velocity_list.index(0.0)
    elif parameter == 'force':
        contexts_list = force_list
        # # default_id = force_list.index(0.001)
    elif parameter == 'gravity':
        contexts_list = gravity_list
        # # default_id = gravity_list.index(0.0025)
    elif parameter == 'start_position':
        contexts_list = start_position_list
        # # default_id = start_position_list.index(-0.5)
    elif parameter == 'start_position_std':
        contexts_list = start_position_std_list
        # # default_id = start_position_std_list.index(0.1)
    elif parameter == 'start_velocity':
        contexts_list = start_velocity_list
        # # default_id = start_velocity_list.index(0.0)
    elif parameter == 'start_velocity_std':
        contexts_list = start_velocity_std_list
        # # default_id = start_velocity_std_list.index(0.0)
    else:
        raise ValueError('Invalid parameter')
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLMountainCar.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLMountainCar.get_default_context()}
    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLMountainCar(context_selector=StaticSelector(contexts=contexts))

    return env, contexts


def generate_bipedalwalkerEnvs(parameter):
    
    fps_list = np.arange(50, 70, 5)
    scale_list = np.arange(30.0, 50.0, 5)
    gravity_x_list = np.arange(0.0, 1.0, 0.1)
    gravity_y_list = np.arange(-10.0, -0.1, 1)
    friction_list = np.arange(2.5, 3.5, 0.5)
    terrain_step_list = np.arange(0.4666666666666667, 0.5666666666666667, 0.02)
    terrain_length_list = np.arange(200, 300, 20)
    terrain_height_list = np.arange(5, 10, 1)
    terrain_grass_list = np.arange(10, 15, 1)
    terrain_startpad_list = np.arange(20, 30, 2)
    motors_torque_list = np.arange(80, 100, 4)
    speed_hip_list = np.arange(4, 6, 0.5)
    speed_knee_list = np.arange(6, 8, 0.5)
    lidar_range_list = np.arange(5.333333333333333, 8.0, 1)
    leg_down_list = np.arange(-0.26666666666666666, -2, -0.5)
    leg_w_list = np.arange(0.26666666666666666, 2, 0.5)
    leg_h_list = np.arange(1.1333333333333333, 2, 0.1)
    initial_random_list = np.arange(5.0, 10.0, 1)
    viewport_w_list = np.arange(600, 800, 40)
    viewport_h_list = np.arange(400, 600, 40)

    if parameter == 'fps':
        contexts_list = fps_list
    elif parameter == 'scale':
        contexts_list = scale_list
    elif parameter == 'gravity_x':
        contexts_list = gravity_x_list
    elif parameter == 'gravity_y':
        contexts_list = gravity_y_list
    elif parameter == 'friction':
        contexts_list = friction_list
    elif parameter == 'terrain_step':
        contexts_list = terrain_step_list
    elif parameter == 'terrain_length':
        contexts_list = terrain_length_list
    elif parameter == 'terrain_height':
        contexts_list = terrain_height_list
    elif parameter == 'terrain_grass':
        contexts_list = terrain_grass_list
    elif parameter == 'terrain_startpad':
        contexts_list = terrain_startpad_list
    elif parameter == 'motors_torque':
        contexts_list = motors_torque_list
    elif parameter == 'speed_hip':
        contexts_list = speed_hip_list
    elif parameter == 'speed_knee':
        contexts_list = speed_knee_list
    elif parameter == 'lidar_range':
        contexts_list = lidar_range_list
    elif parameter == 'leg_down':
        contexts_list = leg_down_list
    elif parameter == 'leg_w':
        contexts_list = leg_w_list
    elif parameter == 'leg_h':
        contexts_list = leg_h_list
    elif parameter == 'initial_random':
        contexts_list = initial_random_list
    elif parameter == 'viewport_w':
        contexts_list = viewport_w_list
    elif parameter == 'viewport_h':
        contexts_list = viewport_h_list
    else:
        raise ValueError('Invalid parameter')
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLBipedalWalker.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLBipedalWalker.get_default_context()}
    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLBipedalWalker(context_selector=StaticSelector(contexts=contexts))

    return env, contexts


def generate_LunarLanderEnvs(parameter):

    FPS_list = [50, 100, 150, 200, 250, 300]
    SCALE_list = [30, 60, 90, 100]
    MAIN_ENGINE_POWER_list = [5, 10, 13, 20, 30, 40, 50]
    SIDE_ENGINE_POWER_list = [0.6, 1, 5, 10, 20, 30, 40, 50]
    INITIAL_RANDOM_list = [100, 500, 1000, 1500, 2000]
    GRAVITY_X_list = [-10, -5, 0.0, 5, 10]
    GRAVITY_Y_list = [-20, -15, -10, -5, 0.0]
    LEG_AWAY_list = [0, 10, 20, 30, 40, 50]
    LEG_DOWN_list = [0, 10, 18, 30, 40, 50]
    LEG_W_list = [2, 5, 8, 10]
    LEG_H_list = [8, 10, 15, 20]
    LEG_SPRING_TORQUE_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    SIDE_ENGINE_HEIGHT_list = [10, 14, 20, 30, 40, 50]
    SIDE_ENGINE_AWAY_list = [10, 12, 20, 30, 40, 50]
    VIEWPORT_W_list = [400, 600, 800, 1000]
    VIEWPORT_H_list =  [200, 400, 600, 800]


    if parameter == 'FPS':
        contexts_list = FPS_list
        # default_id = FPS_list.index(50)
    elif parameter == 'SCALE':
        contexts_list = SCALE_list
        # default_id = SCALE_list.index(30)
    elif parameter == 'MAIN_ENGINE_POWER':
        contexts_list = MAIN_ENGINE_POWER_list
        # default_id = MAIN_ENGINE_POWER_list.index(20)
    elif parameter == 'SIDE_ENGINE_POWER':
        contexts_list = SIDE_ENGINE_POWER_list
        # default_id = SIDE_ENGINE_POWER_list.index(20)
    elif parameter == 'INITIAL_RANDOM':
        contexts_list = INITIAL_RANDOM_list
        # default_id = INITIAL_RANDOM_list.index(10)
    elif parameter == 'GRAVITY_X':
        contexts_list = GRAVITY_X_list
        # default_id = GRAVITY_X_list.index(0.0)
    elif parameter == 'GRAVITY_Y':
        contexts_list = GRAVITY_Y_list
        # default_id = GRAVITY_Y_list.index(-10)
    elif parameter == 'LEG_AWAY':
        contexts_list = LEG_AWAY_list
        # default_id = LEG_AWAY_list.index(20)
    elif parameter == 'LEG_DOWN':
        contexts_list = LEG_DOWN_list
        # default_id = LEG_DOWN_list.index(18)
    elif parameter == 'LEG_W':
        contexts_list = LEG_W_list
        # default_id = LEG_W_list.index(2)
    elif parameter == 'LEG_H':
        contexts_list = LEG_H_list
        # default_id = LEG_H_list.index(8)
    elif parameter == 'LEG_SPRING_TORQUE':
        contexts_list = LEG_SPRING_TORQUE_list
        # default_id = LEG_SPRING_TORQUE_list.index(40)
    elif parameter == 'SIDE_ENGINE_HEIGHT':
        contexts_list = SIDE_ENGINE_HEIGHT_list
        # default_id = SIDE_ENGINE_HEIGHT_list.index(14)
    elif parameter == 'SIDE_ENGINE_AWAY':
        contexts_list = SIDE_ENGINE_AWAY_list
        # default_id = SIDE_ENGINE_AWAY_list.index(12)
    elif parameter == 'VIEWPORT_W':
        contexts_list = VIEWPORT_W_list
        # default_id = VIEWPORT_W_list.index(600)
    elif parameter == 'VIEWPORT_H':
        contexts_list = VIEWPORT_H_list
        # default_id = VIEWPORT_H_list.index(400)
    else:
        raise ValueError('Invalid parameter')
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLLunarLander.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLLunarLander.get_default_context()}

    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLLunarLander(context_selector=StaticSelector(contexts=contexts))

    return env, contexts

def generate_VehicleRacingEnvs(parameter):

    VEHICLE_list = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    
    if parameter == 'VEHICLE':
        contexts_list = VEHICLE_list
        # default_id = VEHICLE_list.index(0)
    
    # generate env contexts
    env_contexts = []
    for c in contexts_list:
        env_context = CARLVehicleRacing.get_default_context().copy()
        env_context[parameter] = c
        env_contexts.append(env_context)

    contexts = {0: CARLVehicleRacing.get_default_context()}

    for i, env_context in enumerate(env_contexts):
        contexts[i+1] = env_context

    env = CARLVehicleRacing(context_selector=StaticSelector(contexts=contexts))

    return env, contexts




def gen_gym(env_name, parameter):
    
    if 'CartPole' in env_name:
        return generate_CartPoleEnvs(parameter)
    elif 'Acrobot' in env_name:
        return generate_AcrobotEnvs(parameter)
    elif 'LunarLander' in env_name:
        return generate_LunarLanderEnvs(parameter)
    elif 'VehicleRacing' in env_name:
        return generate_VehicleRacingEnvs(parameter)
    else:
        raise ValueError(f"Unknown environment: {env_name}")