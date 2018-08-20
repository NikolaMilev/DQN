# The metaparameters can be found at the beginning of the code, below imports.
# Fine-tune them for a different behaviour.

#KERAS
from keras.models import Model
import keras.backend as kback
from keras.layers import Input, Convolution2D, Dense, Flatten, Lambda, Multiply
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import History 

# for resizing
import scipy

#other
from skimage.transform import resize
import numpy as np
from collections import deque
import numpy as np
import gym
import random

import datetime
import os
# to make sure the image data is in the correct order
import keras.backend as backend
assert backend.image_data_format()=="channels_last"

# for frame testing purposes only
#import matplotlib.pyplot as plt

GAME="BreakoutDeterministic-v4" # The name of the environment. See OpenAI Gym site for potential changes. 
COLAB=False # tried working on Google Colaboratory so this is True if this code is being run on that platform. For saving purposes.

TRAIN_FREQUENCY=4       # The network is trained each TRAIN_FREQUENCY time steps.
SAVE_FREQUENCY=25000   	# The network is saved to disk each SAVE_FREQUENCY parameter updates. 
PADDING="valid" 		# No padding.

USE_TARGET_NETWORK=True # If True, target network will be used. Otherwise, it will not be.
SAVE_PATH=os.path.join("colaboratory_models", "colab_models") if COLAB else "."
SAVE_NAME=GAME+str(datetime.datetime.now())

NETWORK_UPDATE_FREQUENCY=10000 # In parameter updates, not in steps taken.

INITIAL_REPLAY_MEMORY_SIZE=TRAIN_FREQUENCY
MAX_REPLAY_MEMORY_SIZE=TRAIN_FREQUENCY # This needed to be reduced for us to simulate no memory.
OBSERVE_MAX=30
NUM_EPISODES = 20000 if COLAB else 50000 # Refers to the number of in-game episodes, not learning episodes. This is here just in case.
TIMESTEP_LIMIT = 10000000	# The duration of the training, in timesteps. 
# one learning episode is separated by loss of life 
MINIBATCH_SIZE=TRAIN_FREQUENCY # Size of the minibatches that are given to the network for one update.
INITIAL_EPSILON=1.0 # Inital exploration rate value.
FINAL_EPSILON=0.1 # Final exploration rate value.
EPSILON_DECAY_STEPS=1000000 # Number of steps over which exlopration rate is annealed from INITIAL_EPSILON to FINAL_EPSILON.
GAMMA=0.99 # Discount rate.
# network details:
NET_H=105 # The height of the image given to the network.
NET_W=80  # The width of the image given to the network.
NET_D=4	  # The number of the images stacked together to make a state.
NET_SIZE="small" # If "small", the smaller architecture will be used (like in Mnih DQN paper). Otherwise, if "large" the larger architecture,
# as in DeepMind paper will be used. Otherwise, a ValueError will be thrown.
# lr je 2.5e-4 u originalnom radu a 5e-5 u novom, poboljsanom
LEARNING_RATE = 2.5e-3 # This learning rate worked best so far. In DeepMind's implementation, they use 2.5e-4 but a different implementation of RMSProp, too.

# RMSProp arguments
MOMENTUM = 0.95  
MIN_GRAD = 0.01

output_path=os.path.join(SAVE_PATH, "training_output.txt")

INFO_WRITE_FREQ=10

TEST_STEPS=10000 		# Timesteps to test.
TEST_FREQ=200000 		# Test frequency in steps. 
TEST_SET=None			# In the beginning, the test set is empty.
TEST_SET_SIZE=2000		# Test set size.
TEST_EPSILON=0.05		# Epsilon (exploration rate) used for testing.
# Utility functions
# I wish to keep this in one file so that I can use it from a notebook

history=History()

def printmsg(msg):
	if(output_path):
			with open(output_path, "a") as out_file:
				out_file.write(msg+"\n")
	else:
		print(msg)

def buildNetwork(height, width, depth, numActions):
	"""
	For creating the network. See NET_SIZE comment.
	"""
	if NET_SIZE == "small":
		return buildNetworkSmall(height, width, depth, numActions)
	elif NET_SIZE == "large":
		return buildNetworkLarge(height, width, depth, numActions)
	else:
		raise ValueError("Unknown parameter for the network size!")


def buildNetworkSmall(height, width, depth, numActions):
	"""
	For building the smaller network, as in Mnih's DQN paper.
	"""
	state_in=Input(shape=(height, width, depth))
	action_in=Input(shape=(numActions, ))
	normalizer=Lambda(lambda x: x/255.0)(state_in)
	conv1=Convolution2D(filters=16, kernel_size=(8,8), strides=(4,4), padding=PADDING, activation="relu")(normalizer)
	conv2=Convolution2D(filters=32, kernel_size=(4,4), strides=(2,2), padding=PADDING, activation="relu")(conv1)
	flatten=Flatten()(conv2)
	dense=Dense(units=256, activation="relu")(flatten)
	out=Dense(units=numActions, activation="linear")(dense)
	filtered_out=Multiply()([out, action_in])
	model=Model(inputs=[state_in, action_in], outputs=filtered_out)
	opt=RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD, clipvalue=1.0) # , clipvalue=1.0
	model.compile(loss="mse", optimizer=opt)

	printmsg("Built and compiled the network!")
	return model

def buildNetworkLarge(height, width, depth, numActions):
	"""
	For building the larger network, as in DeepMind's DQN paper.
	"""
	state_in=Input(shape=(height, width, depth))
	action_in=Input(shape=(numActions, ))
	normalizer=Lambda(lambda x: x/255.0)(state_in)
	conv1=Convolution2D(filters=32, kernel_size=(8,8), strides=(4,4), padding=PADDING, activation="relu")(normalizer)
	conv2=Convolution2D(filters=64, kernel_size=(4,4), strides=(2,2), padding=PADDING, activation="relu")(conv1)
	conv3=Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding=PADDING, activation="relu")(conv2)
	flatten=Flatten()(conv3)
	dense=Dense(units=512, activation="relu")(flatten)
	out=Dense(units=numActions, activation="linear")(dense)
	filtered_out=Multiply()([out, action_in])
	model=Model(inputs=[state_in, action_in], outputs=filtered_out)
	opt=RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD)
	model.compile(loss=LOSS, optimizer=opt)

	printmsg("Built and compiled the network!")
	return model

def copyModelWeights(srcModel, dstModel):
	"""
	Copies the weights from srcModel to dstModel
	"""
	dstModel.set_weights(srcModel.get_weights())

def saveModelWeights(model, name=SAVE_NAME):
	"""
	Saves the weights of the first argument to disk.
	"""
	savePath=os.path.join(SAVE_PATH, name + ".h5")
	model.save_weights(savePath)
	printmsg("Saved weights to {}".format(savePath))


def preprocessSingleFrame(img):
	"""
	Resizes the image by half and extracts the Y channel.
	"""
	# Y = 0.299 R + 0.587 G + 0.114 B
	# with double downsample
	view = img[::2,::2]
	return (view[:,:,0]*0.299 + view[:,:,1]*0.587 + view[:,:,2]*0.114).astype(np.uint8)

	#return preprocessSingleFrameNew(img)

# we will use tuples!
def getNextState(state, nextFrame):
	"""
	Concatenates the new frame to the previous state and discarding the first element of the previoust state,
	thus creating a new state.
	"""
	return (state[1], state[2], state[3], preprocessSingleFrame(nextFrame))

def transformReward(reward):
	"""
	Transforms the reward by clipping it to the interval [-1,1]
	"""
	return np.clip(reward, -1.0, 1.0)
	#return reward

class ExperienceReplay():
	"""
	The class for memory. So far, one tuple is ~10KB
	I used tuples because this way, after preprocessing, up to 4 consecutive
	states share some frame data. When using numpy arrays, this was impossible
	as I was appending data. The only more efficient way is to use one large
	numpy array for all screenshots but this seems overly complicated because
	end of an episode edge case. 
	This version still contains the memory class because it was much easier this way than
	to reimplement the agent. 
	"""
	def __init__(self, maxlen=MAX_REPLAY_MEMORY_SIZE):
		self.memory=deque(maxlen=maxlen)
	def size(self):
		"""
		Returns the current size of the memory.
		"""
		return len(self.memory)
	def addTuple(self, state, action, reward, nextState, terminal):
		global TEST_SET
		self.memory.append((state, action, reward, nextState, terminal))

	def sample(self, sampleSize=MINIBATCH_SIZE):
		return random.sample(self.memory, sampleSize)
	def getMiniBatch(self, sampleSize=MINIBATCH_SIZE):
		minibatch=self.sample(sampleSize) # an array of tuples
		states=np.array([np.stack([frame for frame in tup[0]], axis=2) for tup in minibatch])
		actions=np.array([tup[1] for tup in minibatch])
		rewards=np.array([tup[2] for tup in minibatch])
		nextStates=np.array([np.stack([frame for frame in tup[3]], axis=2) for tup in minibatch])
		terminals=np.array([tup[4] for tup in minibatch])
		
		assert states.dtype==np.uint8
		assert nextStates.dtype==np.uint8
		assert terminals.dtype==bool

		return (states, actions, rewards, nextStates, terminals)




class DQNAgent():
	def __init__(self, envName):
		self.envName=envName
		self.env=gym.make(self.envName)
		self.numActions=self.env.action_space.n
		self.experienceReplay=ExperienceReplay()
		self.qNetwork=buildNetwork(NET_H, NET_W, NET_D, self.numActions)
		if(USE_TARGET_NETWORK):
			self.targetNetwork=buildNetwork(NET_H, NET_W, NET_D, self.numActions)
			copyModelWeights(srcModel=self.qNetwork, dstModel=self.targetNetwork)
		self.bestReward=-1.0

		self.testExperienceReplay=ExperienceReplay(maxlen=TEST_SET_SIZE)

		# actions chosen
		self.timeStep=0
		# episode count
		self.episodeCount=0

		# the initial epsilon (exploration) value
		self.epsilon=INITIAL_EPSILON
		# the value by which epsilon is decreased for every action taken after
		# the INITIAL_REPLAY_MEMORY_SIZEth frame
		self.epsilonDecay=(INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY_STEPS

		self.parameterUpdates=0 # to count the number of parameter updates so I can copy network weights


	def printInfo(self):
		"""
		Printing the current training info.
		"""
		out = "Ep: {}, Dur: {}, Step: {}, Rew: {:.2f}, Loss: {:.4f}, Eps: {:.4f}, Mem.size: {}".format(self.episodeCount, self.episodeDuration, self.timeStep, self.episodeReward, self.episodeLoss, self.epsilon, self.experienceReplay.size())
		printmsg(out)

	def chooseAction(self, state):
		"""
		Choses the action, following the self.epsilon - greedy policy.
		"""
		retval=None
		if self.timeStep < INITIAL_REPLAY_MEMORY_SIZE or np.random.rand() < self.epsilon:
			retval=self.env.action_space.sample()
		else:
			stacked_state=np.stack(state, axis=2)
			y=self.qNetwork.predict([np.expand_dims(stacked_state, axis=0), np.expand_dims(np.ones(self.numActions), axis=0)])
			retval=np.argmax(y, axis=1)
		assert retval!=None

		if self.epsilon > FINAL_EPSILON and self.timeStep >= INITIAL_REPLAY_MEMORY_SIZE:
			self.epsilon-=self.epsilonDecay
		return retval

	def trainOnBatch(self, batchSize=MINIBATCH_SIZE):
		"""
		Training on batch, which is, in this version, 4 states.
		"""
		self.parameterUpdates+=1
		states, actions, rewards, nextStates, terminals=self.experienceReplay.getMiniBatch()
		actions=to_categorical(actions, num_classes=self.numActions)
		if(USE_TARGET_NETWORK):
			nextStateValues=self.targetNetwork.predict([nextStates, np.ones(actions.shape)], batch_size=batchSize)
		else:
			nextStateValues=self.qNetwork.predict([nextStates, np.ones(actions.shape)], batch_size=batchSize)
		assert terminals.dtype==bool
		nextStateValues[terminals]=0
		# 
		y=rewards + GAMMA * np.max(nextStateValues, axis=1)
		y=np.expand_dims(y, axis=1)*actions
		#self.episodeLoss += self.qNetwork.train_on_batch([states, actions], y)

		hist=self.qNetwork.fit([states, actions], y, batch_size=batchSize, epochs=1, verbose=0)
		#self.episodeLoss+=np.mean(hist.history['loss'])

		# if self.parameterUpdates % 1000 == 0:
		# 	printmsg("Parameter Updates: {}".format(self.parameterUpdates))

		if USE_TARGET_NETWORK and (self.parameterUpdates % NETWORK_UPDATE_FREQUENCY == 0):
			copyModelWeights(srcModel=self.qNetwork, dstModel=self.targetNetwork)
			printmsg("Updated target network!")


	def test(self, network, numSteps=TEST_STEPS):
		"""
		Tests the network, as described in Mnih's paper.
		"""
		
		# calculate average max Q for fixed states
		if TEST_SET is None:
			printmsg("Test set None!")
			return
		qs = network.predict([TEST_SET, np.ones((len(TEST_SET),self.numActions))], batch_size=len(TEST_SET))
		qs = np.max(qs, axis=1)

		# make a new environment and test on it for average reward per episode on the target network
		testEnv=gym.make(self.envName)
		testTimeStep=0
		testReward=0
		testEpisode=0
		numActions=testEnv.action_space.n
		duration=0
		for testEpisode in range(TEST_STEPS): # there will be no more than TEST_STEPS episodes, for sure!
			#printmsg("{}".format(testTimeStep))
			if testTimeStep >= TEST_STEPS:
				break
			terminal=False
			observation=testEnv.reset() # return frame

			for _ in range(random.randint(1, OBSERVE_MAX)):
				observation, _, _, info=testEnv.step(0)

			frame=preprocessSingleFrame(observation)
			state=(frame, frame, frame, frame)
			
			while not terminal:
				duration+=1
				# choose an action
				if np.random.rand() < TEST_EPSILON:
					action = testEnv.action_space.sample()
				else:
					stackedState=np.stack(state, axis=2)
					y=network.predict([np.expand_dims(stackedState, axis=0), np.expand_dims(np.ones(numActions), axis=0)])
					action=np.argmax(y, axis=1)
				# send the action to the environment
				observation, reward, terminal, info = testEnv.step(action)
				testReward+=reward
				testTimeStep+=1
				state=getNextState(state, observation)
		testEnv.close()
		meanQs=np.mean(qs)
		avgReward=testReward*1.0/testEpisode
		avgDuration=duration*1.0/testEpisode
		printmsg("Avg Q value: {:.4f} Avg testing reward: {:.4f} Avg duration: {:.4f}".format(meanQs, avgReward, avgDuration))	
		return (meanQs, avgReward, avgDuration)


		
	def learn(self):
		"""
		The main function for learning.
		"""
		self.timeStep=0
		learningReward=0.0
		rewardInterval=100
		for self.episodeCount in range(NUM_EPISODES):
			if self.episodeCount % rewardInterval == 0:
				printmsg("Avg learning reward last {} episodes: {:.4f}".format(rewardInterval, (learningReward*1.0)/rewardInterval))
				learningReward=0.0
			# we quit if training is done
			if self.timeStep >= TIMESTEP_LIMIT:
				break

			terminal=False
			observation=self.env.reset() # return frame

			for _ in range(random.randint(1, OBSERVE_MAX)):
				observation, _, _, info=self.env.step(0)
			curLives=info['ale.lives']
			frame=preprocessSingleFrame(observation)
			state=(frame, frame, frame, frame)
			nextState=None
			while not terminal:
				action=self.chooseAction(state)
				observation, reward, terminal, info = self.env.step(action)
				newLives=info['ale.lives']
				# I found that the loss of life means the end of an episode
				if newLives < curLives:
					terminalToInsert=True
				else:
					terminalToInsert=False

				curLives=newLives
				nextState=getNextState(state, observation)
				# I wish to see the raw reward
				learningReward+=reward
				reward=transformReward(reward)
				global TEST_SET
				self.experienceReplay.addTuple(state, action, reward, nextState, terminalToInsert)
				if self.testExperienceReplay.size() < TEST_SET_SIZE:
					self.testExperienceReplay.addTuple(state, action, reward, nextState, terminalToInsert)
				elif TEST_SET is None:
					sample = random.sample(self.testExperienceReplay.memory, TEST_SET_SIZE)
					TEST_SET = np.array([np.stack([frame for frame in tup[0]], axis=2) for tup in sample])	

				if self.experienceReplay.size() >= INITIAL_REPLAY_MEMORY_SIZE:
					if self.timeStep % TRAIN_FREQUENCY == 0:
						self.trainOnBatch()
						if self.parameterUpdates % SAVE_FREQUENCY == 0:
							if(USE_TARGET_NETWORK):
								saveModelWeights(self.targetNetwork)
							else:
								saveModelWeights(self.qNetwork)

					if self.timeStep % TEST_FREQ == 0:
						(meanQs, avgReward, avgDuration)=self.test(self.targetNetwork if USE_TARGET_NETWORK else self.qNetwork)
						# saving the network that produced the best reward
						if avgReward > self.bestReward:
							self.bestReward=avgReward
							saveModelWeights(self.targetNetwork if USE_TARGET_NETWORK else self.qNetwork, name="best_network")



				self.timeStep+=1
				state=nextState
				if self.timeStep % 50000 == 0:
					printmsg("Step {}".format(self.timeStep))


agent=DQNAgent(GAME)
agent.learn()
