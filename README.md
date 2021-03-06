# DQN

This is an implementation of the DQN algorithm, for my master thesis. The approach used here is a combination of the implementations given [here](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/) and [here](https://arxiv.org/abs/1312.5602). The version with the memory requires about 10-11GB of free memory to run with full 1000000 frame memory.

## Repo structure

* The **code** directory contains the two versions of the code: *with memory* and *without memory*. Using the target network and other metaparameters can be tuned in the code, using the constants at the beginning of the files. This directory also contains the code used for testing the agent (in the file named tester.py ).

* The **results** directory contains pre-trained networks, graphs and videos. The graphs (both in png and pgf formats) of the obtained training results, is inside the subdirectory **graphs** and the video files for two implementations are inside the subdirectory **videos**. There are two videos, both representing the behaviour of the agent with memory and target network. The difference for the agents in the video files is in the network architecture: one (target_memory.mp4) is using the architecture introduced in Mnih's DQN paper and the other (large_target_memory.mp4) uses DeepMind's architecture. The **networks** subdirectory contains the network weight after training 10 million frames.

## Dependencies

The code was written for python 3.5.2. It also uses the following libraries and their dependencies:

* numpy
* Keras
* OpenAI Gym (with Atari wrappers)

All of these are available using pip.

## Contact

If you want to suggest a change or inquire about this code, feel free to contact me via nikola.n.milev@gmail.com.