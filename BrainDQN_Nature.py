import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
EXPLORE = 30000. #训练中最大交互次数 frames over which to anneal epsilon
#FINAL_EPSILON = 0#0.001 # final value of epsilon
#INITIAL_EPSILON = 0#0.01 # starting value of epsilon
FINAL_EPSILON = 0.001#0.001 # final value of epsilon
INITIAL_EPSILON = 0.001#0.01 # starting value of epsilon
#INITIAL_EPSILON = 0.001#0.01 # starting value of epsilon
INITIAL_EPSILON = 0.3#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # 可存储的最大状态转移信息条数
#BATCH_SIZE = 64 # size of minibatch
#UPDATE_TIME = 200
#BATCH_SIZE = 32 # size of minibatch
#UPDATE_TIME = 100
BATCH_SIZE = 256*2 # size of minibatch
OBSERVE = BATCH_SIZE + 10 # 交互OBSERVE次后用神经网络逼近值函数UPDATE_TIME = 1000
UPDATE_TIME = 1000
alpha = 1e-5
try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply

class BrainDQN:

	def __init__(self,actions, imgDim):
		self.imgDim = imgDim
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStepLast = self.timeStep = 0
		
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		# init Q network
		self.stateInput,self.QValue,self.W1,self.b1,self.W2,self.b2 = self.createQNetwork()
		# init Target Q Network  self.QValueT用于记录用计算TD时用到的下一个状态的值函数
		self.stateInputT,self.QValueT,self.W1T,self.b1T,self.W2T,self.b2T = self.createQNetwork()
		self.loss_temp = 0
		self.copyTargetQNetworkOperation = [self.W1T.assign(self.W1),self.b1T.assign(self.b1),self.W2T.assign(self.W2),self.b2T.assign(self.b2)]
		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print ("Successfully loaded:", checkpoint.model_checkpoint_path)
				a = checkpoint.model_checkpoint_path
				self.timeStepLast = int(a.split('-')[-1])
		else:
				print ("Could not find old network weights")


	def createQNetwork(self):
#		# network weights
            self.single_image_units = (20 + 10)*3*5
#            self.single_units = self.single_image_units + self.actions
            self.single_units = self.single_image_units
            self.in_units = self.single_units*3 + self.single_image_units
            in_units = self.in_units
            h1_units = max(10,int(in_units/2))
            o_units = self.actions
            W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
            b1 = tf.Variable(tf.zeros([h1_units]))
            W2 = tf.Variable(tf.zeros([h1_units, o_units]))
            b2 = tf.Variable(tf.zeros([o_units]))
            
            stateInput = tf.placeholder(tf.float32, [None, in_units])
            hidden1 = tf.nn.relu(tf.matmul(stateInput, W1) + b1)
#            hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
            #y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
            QValue = tf.matmul(hidden1, W2) + b2
            return stateInput,QValue,W1,b1,W2,b2

	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
#		self.trainStep = tf.train.AdagradOptimizer(1e-6).minimize(self.cost)
		self.trainStep = tf.train.AdamOptimizer(alpha).minimize(self.cost)


	def trainQNetwork(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
#		print('nextState_batch:',nextState_batch)
#		print('nextState_batch[0]:',nextState_batch[0])
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.loss_temp, _ = self.session.run([self.cost, self.trainStep],feed_dict={self.yInput : y_batch, self.actionInput : action_batch, self.stateInput : state_batch})

		# save network every 100000 iteration
		if self.timeStep % 3000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)
		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

	def setPerception(self,nextObservation,action,reward,terminal):
		newState = np.hstack((self.currentState[self.single_image_units:],nextObservation[0]))
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"
		print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON %0.3f" %(self.epsilon), "loss: %0.5f" %(self.loss_temp))
		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
#		print("self.currentState:",self.currentState)
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = np.zeros(self.actions)
		action_index = 1
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[action_index] = 1 # do nothing
		# change episilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
		return action

	def setInitState(self,observation,action0):
#		self.currentState = np.hstack((observation[0], action0, observation[0], action0, observation[0], action0, observation[0]))
		self.currentState = np.hstack((observation[0], observation[0], observation[0], observation[0]))

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
