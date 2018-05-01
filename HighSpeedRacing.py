import cv2
import sys
sys.path.append("game/")
import HighSpeedRacingGame as game
from BrainDQN_Nature import BrainDQN
import numpy as np
import matplotlib.pyplot as plt
import time
imgDim = [80*1,80*1]
# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (imgDim[0], imgDim[1])), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(imgDim[0],imgDim[1],1))

def HighSpeedRacing():
    # Step 1: init BrainDQN
    actions = 5
    brain = BrainDQN(actions, imgDim)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([0,1,0,0,0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    print(observation0)
#    print('observation0 1:',observation0)
#    observation0 = cv2.cvtColor(cv2.resize(observation0, (imgDim[0],imgDim[1])), cv2.COLOR_BGR2GRAY)
#    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain.setInitState(observation0,action0)   #将observation0复制4份放进BrainDQN的属性self.currentState中

#    isUseExpertData = False
##    isUseExpertData = True
#    if(isUseExpertData == True):
#        filename = "./expertData/observation"
#        actInd = 0
#        observation0 = np.load(filename + str(actInd) + ".npy")
#        plt.imshow(observation0)
#    #    # Step 3.2: run the game
#    #    while 1!= 0:
#        for _ in range(1):
#            actInd = 0
#            for actInd in range(1,2073):
#                actInd += 1
#                action = np.load(filename + "action" + str(actInd) + ".npy")
#                reward = np.load(filename + "reward" + str(actInd) + ".npy")
#                terminal = np.load(filename + "terminal" + str(actInd) + ".npy")
#                nextObservation = np.load(filename + str(actInd) + ".npy")
#                plt.imshow(nextObservation)
#                nextObservation = preprocess(nextObservation)
#                brain.setPerception(nextObservation,action,reward,terminal)
    loss=[]
    plt.figure()
    ind = 0
    # Step 3.2: run the game
    while 1!= 0:
#        time.sleep(0.1)
        action= brain.getAction()
        loss.append(brain.loss_temp)
        ind += 1
        if ind%500==499:
            plt.plot(loss)
            plt.show()
        nextObservation,reward,terminal = flappyBird.frame_step(action)
#        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,terminal)

def main():
    HighSpeedRacing()

if __name__ == '__main__':
    main()
