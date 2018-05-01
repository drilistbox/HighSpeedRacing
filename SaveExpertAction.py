# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np
imgDim = [80*1,80*1]
# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (imgDim[0], imgDim[1])), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(imgDim[0],imgDim[1],1))

def playFlappyBird():
    # Step 1: init BrainDQN
    actions = 5
    brain = BrainDQN(actions, imgDim)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([0,1,0,0,0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (imgDim[0],imgDim[1])), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain.setInitState(observation0)   #将observation0复制4份放进BrainDQN的属性self.currentState中

    filename = "./expertData/observation"
    '''
    第1次试验用，其他全注释
    '''
#    actInd = 0
#    np.save(filename + str(actInd), observation0)
    
    actInd = 1045 #上次记录的最后一个数字
    # Step 3.2: run the game
    while 1!= 0:
#		action = brain.getAction()5
        act = 0
        while(act not in [2,5,8,6,4]):
            act = input("Please intput your action:")
            if(act == ''): continue
            act = int(act)
            
        if(act == 2): action = np.array([1,0,0,0,0])
        if(act == 5): action = np.array([0,1,0,0,0])
        if(act == 8): action = np.array([0,0,1,0,0])
        if(act == 6): action = np.array([0,0,0,1,0])
        if(act == 4): action = np.array([0,0,0,0,1])
       
#        ExpertAct.append(action.tolist())
        nextObservation,reward,terminal = flappyBird.frame_step(action)
        actInd += 1
        np.save(filename + str(actInd), nextObservation)
        np.save(filename + "action" + str(actInd), action)
        np.save(filename + "reward" + str(actInd), reward)
        np.save(filename + "terminal" + str(actInd), terminal)
        
        nextObservation = preprocess(nextObservation)
    #brain.setPerception(nextObservation,action,reward,terminal)

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()