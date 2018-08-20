# testing metaparameters

PATH_TO_NETWORK="dve_velike_mreze_ima_memoriju/rezultati_10kk"
TRAIN_EPSILON=0
RENDER=True
VIDEO_SAVE=True
VIDEO_SAVE_PATH="/home/nmilev/Desktop/openai_video"
NN="big"

#KERAS
from keras.models import Model
import keras.backend as kback
from keras.layers import Input, Convolution2D, Dense, Flatten, Lambda, Multiply
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import History 

import scipy


#other
from skimage.transform import resize
import numpy as np
from collections import deque
import numpy as np
import gym
import random
import time
import datetime
import os
# to make sure the image data is in the correct order
import keras.backend as backend
assert backend.image_data_format()=="channels_last"
from gym import wrappers
# for frame testing purposes only
#import matplotlib.pyplot as plt

GAME="BreakoutDeterministic-v4"
COLAB=False

SAVE_PATH=os.path.join("colaboratory_models", "colab_models") if COLAB else "."
SAVE_NAME=GAME+str(datetime.datetime.now())

LOAD_PATH=os.path.join(PATH_TO_NETWORK, "best_network.h5")

INITIAL_REPLAY_MEMORY_SIZE=50000
MAX_REPLAY_MEMORY_SIZE=1000000 if COLAB else 500000
OBSERVE_MAX=30
NUM_EPISODES = 1
MINIBATCH_SIZE=32

GAMMA=0.99
# network details:
NET_H=105
NET_W=80
NET_D=4
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.95  
MIN_GRAD = 0.01
#LOSS=huberLoss


PADDING="valid"



INFO_WRITE_FREQ=10

# Utility functions
# I wish to keep this in one file so that I can use it from a notebook

history=History()


LOSS="mse"

def buildNetwork(height, width, depth, numActions):
	if NN == "big":
		return buildNetworkBig(height, width, depth, numActions)
	else:
		return buildNetworkSmall(height, width, depth, numActions)

def buildNetworkSmall(height, width, depth, numActions):
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
	model.compile(loss=LOSS, optimizer=opt)

	print("Built and compiled the network!")
	return model

def buildNetworkBig(height, width, depth, numActions):
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
	opt=RMSprop(lr=LEARNING_RATE, rho=MOMENTUM, epsilon=MIN_GRAD, clipvalue=1.0)
	model.compile(loss=LOSS, optimizer=opt)

	print("Built and compiled the network!")
	return model

def saveModelWeights(model):
	savePath=os.path.join(SAVE_PATH, SAVE_NAME + ".h5")
	model.save_weights(savePath)
	print("Saved weights to {}".format(savePath))

def loadModelWeights(model, path):
	model.load_weights(path)

def getModel(path, height, width, depth, numActions):
	model=buildNetwork(height, width, depth, numActions)
	loadModelWeights(model, path)
	return model

def preprocessSingleFrameNew(img):
	view=img
	#view=img[::2,::2]
	x=(view[:,:,0]*0.299 + view[:,:,1]*0.587 + view[:,:,2]*0.114)
	p=scipy.misc.imresize(x, (84, 84)).astype(np.uint8)
	# plt.imshow(p)
	# plt.show()
	return p

def preprocessSingleFrame(img):
	# Y = 0.299 R + 0.587 G + 0.114 B
	# with double downsample
	view = img[::2,::2]
	return (view[:,:,0]*0.299 + view[:,:,1]*0.587 + view[:,:,2]*0.114).astype(np.uint8)
	#return preprocessSingleFrameNew(img)

# we will use tuples!
def getNextState(state, nextFrame):
	return (state[1], state[2], state[3], preprocessSingleFrame(nextFrame))

def transformReward(reward):
	return np.clip(reward, -1.0, 1.0)


class DRLAgent():
	def __init__(self, envName):
		self.envName=envName
		self.env=gym.make(self.envName)
		if VIDEO_SAVE:
			self.env=wrappers.Monitor(self.env, VIDEO_SAVE_PATH, force=True, video_callable=lambda episode_id: True)
		self.numActions=self.env.action_space.n
		self.qNetwork=getModel(LOAD_PATH, NET_H, NET_W, NET_D, self.numActions)

		# actions chosen
		self.timeStep=0
		# episode count
		self.episodeCount=0
		# total episode reward
		self.episodeReward=0.0
		# total episode loss; has nothing to do with reward as the loss
		# is obtained by training on random batches
		self.episodeLoss=0.0
		# the total episode duration in frames, including no-op frames
		self.episodeDuration=0

		# the initial epsilon (exploration) value
		self.epsilon=TRAIN_EPSILON
		# the value by which epsilon is decreased for every action taken after
		# the INITIAL_REPLAY_MEMORY_SIZEth frame


	def printInfo(self):
		print("Ep: {}, Dur: {}, Step: {}, Rew: {:.2f}".format(self.episodeCount, self.episodeDuration, self.timeStep, self.episodeReward, self.episodeLoss, self.epsilon))

	def chooseAction(self, state):
		retval=None
		if np.random.rand() < self.epsilon:
			retval=self.env.action_space.sample()
		else:
			stacked_state=np.stack(state, axis=2)
			y=self.qNetwork.predict([np.expand_dims(stacked_state, axis=0), np.expand_dims(np.ones(self.numActions), axis=0)])
			retval=np.argmax(y, axis=1)
		assert retval!=None

		return retval
		
	def run(self, numEpisodes=NUM_EPISODES):
		print(self.qNetwork.summary())
		self.timeStep=0
		for self.episodeCount in range(numEpisodes):
			self.episodeDuration=0
			self.episodeLoss=0
			self.episodeReward=0
			self.episodeDuration=0

			terminalToInsert=False
			terminal=False
			observation=self.env.reset() # return frame

			for _ in range(random.randint(1, OBSERVE_MAX)):
				observation, _, _, info=self.env.step(0)
				self.episodeDuration += 1
				if RENDER:
					self.env.render()
				time.sleep(1/15.0)
			curLives=info['ale.lives']
			frame=preprocessSingleFrame(observation)
			state=(frame, frame, frame, frame)
			nextState=None
			while not terminal:
				if terminalToInsert:
					action=1
					print("MORTUS")
				else:
					action=self.chooseAction(state)

				observation, reward, terminal, info = self.env.step(action)
				newLives=info['ale.lives']
				# I found that the loss of life means the end of an episode
				if newLives < curLives:
					terminalToInsert=True
				else:
					terminalToInsert=False
				curLives=newLives
				#print(type(observation))
				nextState=getNextState(state, observation)
				# I wish to see the raw reward
				self.episodeReward+=reward
				reward=transformReward(reward)
				if RENDER:
					self.env.render()
				self.timeStep+=1
				self.episodeDuration += 1
				time.sleep(1/60.0)
				
				
				state=nextState

				self.printInfo()
		self.env.close()

agent=DRLAgent(GAME)
agent.run()
