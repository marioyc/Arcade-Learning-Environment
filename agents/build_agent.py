"""
:description: training constants and building training agents
"""

import common.file_utils
import common.feature_extractors as feature_extractors
import learning_agents

###  parameters common to all agents ###
ACTIONS = [0,1,3,4]
DISCOUNT = .99#.993
EXPLORATION_PROBABILITY = 0.05#0.01#1
STEP_SIZE = 0.01#0.5 #.001
MAX_GRADIENT = 5
NUM_CONSECUTIVE_RANDOM_ACTIONS = 0 # 0 denotes only taking a random action once
FEATURE_EXTRACTOR = feature_extractors.BasicFeature() #TrackingClassifyingContourExtractor()
########################################

###### feature extractor options: ######
# OpenCVBoundingBoxExtractor()
# TrackingClassifyingContourExtractor()
########################################

########################################
def build_sarsa_agent():
    print 'building sarsa agent...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.SARSALearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE,
                maxGradient=MAX_GRADIENT,
                num_consecutive_random_actions=NUM_CONSECUTIVE_RANDOM_ACTIONS)
########################################

######## sarsa lambda parameters #######
THRESHOLD = .1
DECAY = 0.9 * DISCOUNT #.98
########################################
def build_sarsa_lambda_agent():
    print 'building sarsa lambda agent...'
    featureExtractor = FEATURE_EXTRACTOR
    return learning_agents.SARSALambdaLearningAlgorithm(
                actions=ACTIONS,
                discount=DISCOUNT,
                featureExtractor=featureExtractor,
                explorationProb=EXPLORATION_PROBABILITY,
                stepSize=STEP_SIZE,
                threshold=THRESHOLD,
                decay=DECAY,
                maxGradient=MAX_GRADIENT,
                num_consecutive_random_actions=NUM_CONSECUTIVE_RANDOM_ACTIONS)
########################################

def load_agent_weights(agent, weights_filepath):
    pass
