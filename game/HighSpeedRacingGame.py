import numpy as np
import sys
import random
import pygame
import utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 30
patchNumX = 120
STREETNUM = 7
carWidth = 18
carLength = 36
safeForwardPatches = 3
PatcheWidth = 6
PatcheHeight = 6
safeLen = safeForwardPatches*PatcheWidth
SCREENWIDTH  = patchNumX*PatcheWidth
SCREENHEIGHT = STREETNUM*(PatcheHeight*2 + carWidth)
SCREENHEIGTHWithWord = SCREENHEIGHT + 50
LANESIDE = 1
STREETPATCHNUM = 5
FORWARDDECT = 20
BACKWARDDECT = 10

MinObstacleVelX = 2
MaxObstacleVelX = 10
MinPlayerVelX = 1
MaxPlayerVelX = 15
pygame.init()
FPSCLOCK = pygame.time.Clock()
#SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGTHWithWord))
pygame.display.set_caption('High speed racing')

IMAGES, SOUNDS, HITMASKS = utils.load()
#p = 0.4
p = 0.05
p = 0.2
PLAYER_LENGTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
OBSCALE_LENGTH = IMAGES['pipe'][0].get_width()
OBSCALE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
BASEY = SCREENHEIGHT * 1.0
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

centerliney = [(ind+0.5)*SCREENHEIGHT/STREETNUM for ind in range(STREETNUM)]

class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH
        self.totalRunLength = 0
        self.totalTime = 0
        '''
        初始player_car位置设置
        '''
        self.initialVecx = 4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerVelX    =  self.initialVecx    # player's velocity along Y, default same as playerFlapped
#        self.playerVelY    =  SCREENHEIGHT/STREETNUM    # player's velocity along Y, default same as playerFlapped
        self.playerx = int(SCREENWIDTH * 0.15)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)

        '''
        初始obscale_car个数、位置设置
        '''
        self.streetLine = [[] for _ in range(STREETNUM)]
#        self.StreetVelX = [8,6,5,6,5,5,8]
        self.StreetVelX = [3,5,8,3,6,5,4]
        self.carVec = [[] for _ in range(STREETNUM)]
        carNums = [2,3,2,2,3,3,2]
        for streetInd in range(STREETNUM):
            self.streetLine[streetInd] = []
            self.carVec[streetInd] = np.tile([self.StreetVelX[streetInd]], carNums[streetInd]).astype(np.float)
            x = ((np.arange(carNums[streetInd]) + 0.5 + 0.2*(np.random.rand(carNums[streetInd]) - 1))*SCREENWIDTH/carNums[streetInd]).astype(np.int16)
            if(streetInd == 3):
                x = (self.playerx + 2*carLength) + ((np.arange(carNums[streetInd]) + 0.5*(np.random.rand(carNums[streetInd])))*(SCREENWIDTH - (self.playerx + 2*carLength))/carNums[streetInd]).astype(np.int16)
            x.sort()
            for indCar in range(carNums[streetInd]):
                self.streetLine[streetInd].append({'x': x[indCar], 'y': centerliney[streetInd] - OBSCALE_HEIGHT/2})
            
    def frame_step(self, input_actions):
        pygame.event.pump()
        terminal = False
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        '''
        动作设置
        '''
        if input_actions.argmax() == 0:#向上加速
#            self.playerVelY = 5
            self.playerVelY = PatcheHeight
            reward = 0
        if input_actions.argmax() == 1:#do nothing
            self.playerVelX += 0
            self.playerVelY = 0
            reward = 0.1
        if input_actions.argmax() == 2:#向上加速
#            self.playerVelY = -5
            self.playerVelY = -PatcheHeight
            reward = 0
        if input_actions.argmax() == 3:#向前加速
            self.playerVelX += 1
        if input_actions.argmax() == 4:#向后加速
            self.playerVelX -= 1
            
        '''
        防止playerCar撞上前面的obscaleCar
        '''
#        indStart = int(self.playery/(SCREENHEIGHT/STREETNUM))
#        indEnd = int((self.playery + PLAYER_HEIGHT)/(SCREENHEIGHT/STREETNUM))
##        print('indStart:',indStart,'indEnd:',indEnd)
#        for ind in range(indStart,indEnd+1):
#            print('ind:',ind)
#            n = len(self.streetLine[ind])
##            indFront = -1
##            xFront = -1
#            for i2 in range(n):
#                if self.streetLine[ind][(i2)%n]['x'] > self.playerx + OBSCALE_LENGTH + safeLen :
#                    d = self.streetLine[ind][(i2)%n]['x'] - (self.playerx + OBSCALE_LENGTH + safeLen) # 距离前车的距离
#                else:
#                    d = SCREENWIDTH + self.streetLine[ind][(i2)%n]['x'] + OBSCALE_LENGTH + safeLen - self.playerx
#                if self.playerVelX > d:
#                    print('sdfsfsfdsdfsdfsdf,ind:',ind)
#                    self.playerVelX = self.carVec[ind][(i2)%n]
#                print("self.playerVelX:",self.playerVelX)

        self.playerVelX = max(self.playerVelX, MinPlayerVelX)
        self.playerVelX = min(self.playerVelX, MaxPlayerVelX)
        self.baseVelX = -self.playerVelX
        self.totalRunLength += self.playerVelX
        self.playery += self.playerVelY
        '''
        playerCar加速减速奖励
        '''
        if input_actions.argmax() == 3:#向前加速
#            reward = 0.2*(self.playerVelX - 0)
#            reward = 0.1*(self.playerVelX - 4)
            reward = 0.1*(self.playerVelX - MinPlayerVelX)
        if input_actions.argmax() == 4:#向后加速
            reward = 0.1
#            reward = 0.05*(self.playerVelX - MinPlayerVelX)
        #离中心线越近奖励越多    
#        reward = 0.1 + 0.5*abs((int(self.playery + PLAYER_HEIGHT/2))%(int(SCREENHEIGHT/STREETNUM)) - int(SCREENHEIGHT/STREETNUM/2)) + 0.3*abs(self.playerVelX - self.initialVecx)
#        reward = 0.1 + 0.5*abs((int(self.playery + PLAYER_HEIGHT/2))%(int(SCREENHEIGHT/STREETNUM)) - int(SCREENHEIGHT/STREETNUM/2)) + 10*(self.playerVelX - self.initialVecx)
#        reward = 0.1*(self.playerVelX - self.initialVecx)
#        if (abs((int(self.playery + PLAYER_HEIGHT/2))%(int(SCREENHEIGHT/STREETNUM)) - int(SCREENHEIGHT/STREETNUM/2))) <= 5 :
#            reward += 0.1

        '''
        超车数量
        '''
        playerFront = self.playerx + PLAYER_LENGTH
        for ind in range(STREETNUM):
            for obscale in self.streetLine[ind]:
                obscaleCar = obscale['x'] + OBSCALE_LENGTH
#                if obscaleCar <= playerFront <= obscaleCar + 1.1*self.StreetVelX[ind]:
#                if playerFront <= obscaleCar and playerFront + self.StreetVelX[ind] > obscaleCar:
                if playerFront <= obscaleCar and playerFront + self.playerVelX > obscaleCar:
                    self.score += 1
                    #SOUNDS['point'].play()
#                    reward += 1
        self.totalTime += 1
        showStr = [u"    Vec:%3.2fKm/h    AverVec:%3.2f Km/h" %(10*self.playerVelX, 10*self.totalRunLength/self.totalTime),
                   u"Mileage:%5.2fKm      CarPassed:%d" %(self.totalRunLength*10/3600, self.score)]

        '''
        车道线移动速度
        '''
#        self.basex = -((-self.basex + 100) % self.baseShift)
        self.basex = self.basex + self.baseVelX
        if(abs(self.basex + self.baseShift) < abs(max(self.StreetVelX)*2)): self.basex = 0

        '''
        obscaleCar移动速度
        '''
        for ind in range(STREETNUM):
            n = len(self.streetLine[ind])
            for i2 in range(n):
                if self.streetLine[ind][(i2+1)%n]['x'] > self.streetLine[ind][i2]['x'] + OBSCALE_LENGTH + safeLen:
                    d = self.streetLine[ind][(i2+1)%n]['x'] - (self.streetLine[ind][i2]['x'] + OBSCALE_LENGTH + safeLen) # 距离前车的距离
                else:
                    d = SCREENWIDTH + self.streetLine[ind][(i2+1)%n]['x'] + OBSCALE_LENGTH + safeLen - self.streetLine[ind][i2]['x'] 
                if self.carVec[ind][i2] < d:
                    if np.random.rand() > p:
#                        self.carVec[ind][i2] += 0.1
                        self.carVec[ind][i2] += 0.5
                    else:
#                        self.carVec[ind][i2] -= 0.1
                        self.carVec[ind][i2] -= 0.5
                else:
#                    print(d)
                    self.carVec[ind][i2] = self.carVec[ind][(i2+1)%n]

        for ind in range(STREETNUM):
            for indCarVeec in range(len(self.streetLine[ind])):
                self.carVec[ind][indCarVeec] = min(self.carVec[ind][indCarVeec], MaxObstacleVelX)
                self.carVec[ind][indCarVeec] = max(self.carVec[ind][indCarVeec], MinObstacleVelX)

        '''
        限制obscaleCar速度以免撞上playerCar
        '''
        ind = int(self.playery/(SCREENHEIGHT/STREETNUM))
        n = len(self.streetLine[ind])
        for i2 in range(n):
            if self.playerx > self.streetLine[ind][(i2)%n]['x'] + OBSCALE_LENGTH + safeLen:
                d = self.playerx - (self.streetLine[ind][(i2)%n]['x'] + OBSCALE_LENGTH + safeLen) # 距离前车的距离
            else:
                d = SCREENWIDTH + self.playerx + OBSCALE_LENGTH + safeLen - self.streetLine[ind][(i2)%n]['x']
            if self.carVec[ind][(i2)%n] > d:
                self.carVec[ind][(i2)%n] = self.playerVelX
                
        '''
        obscaleCar在屏幕中的位置
        '''
        for ind in range(STREETNUM):
            for indCarVeec in range(len(self.streetLine[ind])):
                self.streetLine[ind][indCarVeec]['x'] += self.carVec[ind][indCarVeec] - self.playerVelX

        for ind in range(STREETNUM):
            if self.streetLine[ind][0]['x'] < -OBSCALE_LENGTH:
                temp = self.streetLine[ind].pop(0)
                temp['x'] += SCREENWIDTH+ OBSCALE_LENGTH
                self.streetLine[ind].append(temp)
            if self.streetLine[ind][-1]['x'] > SCREENWIDTH*1:
                temp = self.streetLine[ind].pop(-1)
                temp['x'] -= SCREENWIDTH + OBSCALE_LENGTH
                self.streetLine[ind].insert(0, temp)

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,'index': self.playerIndex},self.streetLine)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -3

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for ind in range(STREETNUM):
            for obscaleCar in self.streetLine[ind]:
                SCREEN.blit(IMAGES['pipe'][0], (obscaleCar['x'], obscaleCar['y']))

        for ind in range(STREETNUM-1):
            SCREEN.blit(IMAGES['base'], (self.basex, (ind+1)*SCREENHEIGHT/STREETNUM))
        # print score so player overlaps the score
        showScore(self.score, showStr)
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        totalPatches = np.zeros((STREETNUM*STREETPATCHNUM, patchNumX))
        for streetInd in range(STREETNUM):
            for car in self.streetLine[streetInd]:
                carLocxInd = int(car['x']/PatcheWidth)
                totalPatches[(streetInd*STREETPATCHNUM+1):(streetInd*STREETPATCHNUM+4), carLocxInd:int((car['x'] + carLength)/PatcheWidth + 1)] = 1
        yInd = int(self.playery/(PatcheHeight))
        xInd = int(self.playerx/(PatcheWidth))
        totalPatches[yInd:int((self.playery+carWidth)/PatcheHeight), xInd:int((self.playerx+carLength)/PatcheWidth + 1)] = 2
        a = np.vstack((np.ones((STREETNUM*STREETPATCHNUM, patchNumX)),totalPatches))
        totalPatches = np.vstack((a,np.ones((STREETNUM*STREETPATCHNUM, patchNumX))))
        xInd0 = int(STREETNUM*STREETPATCHNUM + yInd - LANESIDE*STREETPATCHNUM - STREETPATCHNUM/2 + 1)
        xInd1 = int(STREETNUM*STREETPATCHNUM + yInd + 1 + LANESIDE*STREETPATCHNUM + STREETPATCHNUM/2 )
        yInd0 = int(xInd + 4 - BACKWARDDECT)
        yInd1 = int(xInd + 4 + FORWARDDECT)
        image_data = totalPatches[xInd0:xInd1, yInd0:yInd1]
        image_data = (image_data).reshape(1, image_data.shape[0]*image_data.shape[1])
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    pipeX = SCREENWIDTH + 10

    return [
#        {'x': pipeX, 'y': centerliney[int(random.random()*(STREETNUM))] - OBSCALE_HEIGHT/2},  # upper pipe
        {'x': pipeX, 'y': centerliney[int(random.random()*(STREETNUM))] - OBSCALE_HEIGHT/2}  # lower pipe
    ]

def showScore(score, showStr):
    """displays score in center of screen"""
#    scoreDigits = [int(x) for x in list(str(score))]
#    totalWidth = 0 # total width of all numbers to be printed
#    for digit in scoreDigits:
#        totalWidth += IMAGES['numbers'][digit].get_width()
#    Xoffset = (SCREENWIDTH - totalWidth) / 2
#    for digit in scoreDigits:
#        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
#        Xoffset += IMAGES['numbers'][digit].get_width()
        
    #获取系统字体，并设置文字大小  
    cur_font = pygame.font.SysFont("Times New Roman", 25)  
    #设置是否加粗属性  
    cur_font.set_bold(False)  
    #设置是否斜体属性  
    cur_font.set_italic(False)  
    #设置文字内容  
#    text = u"car passed:"
    text_fmt1 = cur_font.render(showStr[0], 1, (0, 0, 0))  
    text_fmt2 = cur_font.render(showStr[1], 1, (0, 0, 0))  
    #绘制文字  
    SCREEN.blit(text_fmt1, (20, SCREENHEIGHT + 0))
    SCREEN.blit(text_fmt2, (20, SCREENHEIGHT + 25))

def checkCrash(player, streetLine):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    elif player['y'] <= 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        for ind in range(STREETNUM):
            for uPipe in streetLine[ind]:
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], OBSCALE_LENGTH, OBSCALE_HEIGHT)
                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                if uCollide:return True
    return False

def isChangeStreet(player, streetLine):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    elif player['y'] <= 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        for ind in range(STREETNUM):
            for uPipe in streetLine[ind]:
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], OBSCALE_LENGTH, OBSCALE_HEIGHT)
                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                if uCollide:return True
    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
