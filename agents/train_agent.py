"""
:description: train an agent to play a game
"""
import os
import sys
import time
import numpy as np
import cv2

# atari learning environment imports
from ale_python_interface import ALEInterface

# custom imports
import common.build_agent as build_agent
import common.file_utils as file_utils
import common.screen_utils as screen_utils

######## training parameters #########
FEATURE_EXTRACTOR = feature_extractors.BasicFeature()
LEARNING_ALGORITHM = build_agent.build_sarsa_lambda_agent(FEATURE_EXTRACTOR)
NUM_EPISODES = 5000#3300
EXPLORATION_REDUCTION_AMOUNT = 0#.0005
MINIMUM_EXPLORATION_EPSILON = 0#.1
NUM_FRAMES_TO_SKIP = 4
######################################

########## training options ##########
LOAD_WEIGHTS = False
LOAD_WEIGHTS_FILENAME = ''
DISPLAY_SCREEN = False
PRINT_TRAINING_INFO_PERIOD = 10
NUM_EPISODES_AVERAGE_REWARD_OVER = 100
RECORD_WEIGHTS = True
RECORD_WEIGHTS_PERIOD = 25
######################################

def train_agent(gamepath, agent, n_episodes, display_screen, record_weights,
        reduce_exploration_prob_amount, n_frames_to_skip):
    """
    :description: trains an agent to play a game

    :type gamepath: string
    :param gamepath: path to the binary of the game to be played

    :type agent: subclass RLAlgorithm
    :param agent: the algorithm/agent that learns to play the game

    :type n_episodes: int
    :param n_episodes: number of episodes of the game on which to train
    """

    # load the ale interface to interact with
    ale = ALEInterface()
    ale.setInt('random_seed', 42)

    # display/recording settings, doesn't seem to work currently
    #recordings_dir = './recordings/breakout/'
    # previously "USE_SDL"
    if display_screen:
        if sys.platform == 'darwin':
            print 'darwin'
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX
            #ale.setString("record_screen_dir", recordings_dir);
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
        ale.setBool('display_screen', True)

    ale.loadROM(gamepath)
    ale.setInt("frame_skip", n_frames_to_skip)

    screen_preprocessor = screen_utils.RGBScreenPreprocessor()
    screen_dims = ale.getScreenDims();
    print screen_dims

    rewards = []
    best_reward = 0
    print('starting training...')
    for episode in xrange(n_episodes):
        action = 0
        reward = 0
        newAction = None

        total_reward = 0
        counter = 0
        lives = ale.lives()

        screen = np.zeros((160 * 210), dtype=np.int8)#np.zeros((32, 32, 3), dtype=np.int8)
        state = { "screen" : screen,
                #"objects" : None,
                #"prev_objects": None,
                #"prev_action": 0,
                "action": 0 }
        if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
            video = cv2.VideoWriter('video/episode-{}-{}-video.avi'.format(episode, agent.name), cv2.VideoWriter_fourcc('M','J','P','G'), 24, screen_dims)

        start = time.time()

        while not ale.game_over():
            # if newAction is None then we are training an off-policy algorithm
            # otherwise, we are training an on policy algorithm
            if newAction is None:
                action = agent.getAction(state)
            else:
                action = newAction
            reward += ale.act(action)

            if ale.lives() < lives:
              lives = ale.lives()
              #reward -= 1
            total_reward += reward

            new_screen = ale.getScreen()#getScreenRGB()
            #print screen.shape, new_screen.shape
            if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
                video.write(ale.getScreenRGB())
            #new_screen = screen_preprocessor.preprocess(new_screen)
            new_state = {"screen": new_screen,
                        #"objects": None,
                        #"prev_objects": state["objects"],
                        #"prev_action": state["action"],
                        "action": action}
            if counter % (n_frames_to_skip + 1) == 0:
                newAction = agent.incorporateFeedback(state, action, reward, new_state)

            state = new_state
            reward = 0
            counter += 1

        end = time.time()
        rewards.append(total_reward)

        if agent.explorationProb > MINIMUM_EXPLORATION_EPSILON:
            agent.explorationProb -= reduce_exploration_prob_amount

        print('episode: {}, score: {}, number of frames: {}, time: {:.4f}m'.format(episode, total_reward, counter, (end - start) / 60))

        if total_reward > best_reward and record_weights:
            best_reward = total_reward
            print("Best reward: {}".format(total_reward))

        if episode % PRINT_TRAINING_INFO_PERIOD == 0:
            print '\n############################'
            print '### training information ###'
            print("Average reward: {}".format(np.mean(rewards)))
            print("Last 50: {}".format(np.mean(rewards[-NUM_EPISODES_AVERAGE_REWARD_OVER:])))
            print("Exploration probability: {}".format(agent.explorationProb))
            #print('action: {}'.format(action))
            print('size of weights dict: {}'.format(len(agent.weights)))
            #print('current objects: {}'.format(state['objects']))
            #print('previous objects: {}'.format(state['prev_objects']))
            weights = [v for k,v in agent.weights.iteritems()]
            min_feat_weight = min(weights)
            max_feat_weight = max(weights)
            avg_feat_weight = np.mean(weights)
            print('min feature weight: {}'.format(min_feat_weight))
            print('max feature weight: {}'.format(max_feat_weight))
            print('average feature weight: {}'.format(avg_feat_weight))
            print '############################'
            print '############################\n'

        if episode != 0 and episode % RECORD_WEIGHTS_PERIOD == 0 and record_weights:
            file_utils.save_rewards(rewards, filename='{}-rewards'.format(agent.name))
            file_utils.save_weights(agent.weights, filename='episode-{}-{}-weights'.format(episode, agent.name))
            video.release()

        ale.reset_game()
    return rewards

if __name__ == '__main__':
    game = 'alien.bin'
    gamepath = os.path.join('roms', game)
    agent = LEARNING_ALGORITHM
    ale = ALEInterface()
    ale.loadROM(gamepath)
    actions = ale.getMinimalActionSet()
    agent.actions = actions;
    print agent.actions
    if LOAD_WEIGHTS:
        agent.weights = file_utils.load_weights(WEIGHTS_FILENAME)
    rewards = train_agent(gamepath, agent,
                        n_episodes=NUM_EPISODES,
                        display_screen=DISPLAY_SCREEN,
                        record_weights=RECORD_WEIGHTS,
                        reduce_exploration_prob_amount=EXPLORATION_REDUCTION_AMOUNT,
                        n_frames_to_skip=NUM_FRAMES_TO_SKIP)
