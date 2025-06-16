# Continuous-Control-RL
This is a solution for the Deep Reinforcement Learning nanodegree project "Continuous Control" from Udacity

The code used in this solution is adopted from the Udacity repository below, and I have included the MIT license in all files that I have modified.

# Repository files:

# Environment:
The environment is provided by Unity with the name "reacher", in this environment a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to the torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This project solves the second Version of the reacher environment, which contains 20 identical agents, each with its own copy of the environment.

In each episode, each agent's score is calculated separately by summing all rewards it gets without discounting, and then the average of all agents is calculated and considered to be the total score of the episode.
The environment is considered solved when the 100-episode moving average of agents' average scores is at least +30

To download the Unity Reacher environment:
- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip


# DDPG:
DDPG or Deep Deterministic Policy Gradient is the algorithem used in this project to solve the enviroment, DDPG is a different kind of actor-critic method, and it could be seen as an approximate DQN instead of an actual actor-critic, because the critic in DDPG is used to approximate the maximizer over the Q values of the next state and not as a baseline.
One of the limitations of the DQN agent is that it is not straightforward to use in continuous action spaces
In DDPG we use two deep neural networks, we call one the actor and the other one is the critic:
-  Actor: takes the state S as input, and outputs the optimal policy deterministically, the output is the best believed action for any given state unlike the discrete case where the output is a probability distribution over all possible actions, the actor is basically learning the argmax Q(s,a)
-  Critic: takes the input state S, then it tries to calculate the optimal action-value function by using the actor's best-believed action
-  
Two interesting aspects of DDPG are:
- The use of a replay buffer
- Soft updates to the target networks:
    - In DQN you have two copies of the network weights: the regular network and the target network, where the target network gets a big update every n-step by simply copying the weights of the regular network into the target network
    - IN DDPG, you have two copies of the network weights for each network, a regular for the critic an irregular for the critic, a target for the actor, and a target for the critic, but in DDPG the target networks are updated using a soft updates strategy which is slowly blending the regular network weights with the target network weights, so every time step you mix in 0.01% of regular network weights with target network weights
 
# DDPG Chosen Hyperparameters:
- Replay buffer size: 100000 
- Minibatch size: 128
- Discount factor: 0.99
- TAU for soft update of target parameters: 0.001
- Learning rate of the actor: 0.001
- Learning rate of the critic: 0.001

# Neural Networks Architecture:
- Actor Network consists of:
    - Fully connected layer which takes the state with the size 33, and outputs 256
    - Fully connected layer which takes the ReLU of the output of the previous layer and outputs 4 logits, each representing an action
    - The output from the previous layer is passed through a tanh to ensure the output actions are between -1 and 1 

- Critc Network consists of:
    - Fully connected layer which takes the state with the size 33, and outputs 256
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer along the actions chosen by the actor with size 260 and outputs 256
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer and outputs 128
    - Fully connected layer which takes the Leaky ReLU of the output of the previous layer and outputs 1 logit which resembles the State-Value


