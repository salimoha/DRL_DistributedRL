# Train Two RL Agents to Play Tennis
##### &nbsp;

## Background
For artificial intelligence (AI) to reach its full potential, AI systems need to interact safely and efficiently with humans, as well as other agents. There are already environments where this happens on a daily basis, such as the stock market. And there are future applications that will rely on productive agent-human interactions, such as self-driving cars and other autonomous vehicles.

One step along this path is to train AI agents to interact with other agents in both cooperative and competitive settings. Reinforcement learning (RL) is a subfield of AI that's shown promise. However, thus far, much of RL's success has been in single agent domains, where building models that predict the behavior of other actors is unnecessary. As a result, traditional RL approaches (such as Q-Learning) are not well-suited for the complexity that accompanies environments where multiple agents are continuously interacting and evolving their policies.

##### &nbsp;

## Goal
The goal of this project is to train two RL agents to play tennis. As in real tennis, the goal of each player is to keep the ball in play. And, when you have two equally matched opponents, you tend to see fairly long exchanges where the players hit the ball back and forth over the net.

##### &nbsp;

## The Environment
We'll work with an environment that is similar, but not identical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-Agents GitHub page.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

![Trained Agent][image1]

##### &nbsp;


## Plot of rewards
![Reward Plot](https://github.com/salimoha/DRL_DistributedRL/blob/master/result.png?raw=true)


##### &nbsp;
## Methodology
We have three main steps to build an agent to work in the above environment as below:

1. Perform random action policy to consider a baseline approach.
2. Choose an appropriate distrubted RL algorithm such as Multi-Agent Deep Deterministic Policy Gradient (MADDPG).
3. Run different experiments until the performance threshold is reached.

### 1. Establish Baseline
Before building agents that learns, I started by testing ones that select actions (uniformly) at random at each time step.

Running the random agents a few times resulted in scores from 0 to 0.02. Obviously, if these agents need to achieve an average score of 0.5 over 100 consecutive episodes, then choosing actions at random won't work. However, when you watch the agents acting randomly, it becomes clear that these types of sporadic actions can be useful early in the training process. That is, they can help the agents explore the action space to find some signal of good vs. bad actions. This insight will come into play later when we implement the Ornstein-Uhlenbeck process and epsilon noise decay.

### 2. Implement Learning Algorithm
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Tennis environment.

#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.



## Future Improvements
- **Address stability issues to produce more consistent results** &mdash; My "best" results are only reproducible if you run the model numerous times. If you just run it once (or even 3-5 times) the model might not converge. I ran the model at least 30 while searching for a good set of hyperparameters, so perhaps implementing a more systemic approach such as grid search would help. Otherwise, more research is needed to find a more stable algorithm, or to make changes to the current DDPG algorithm.
- **Add *prioritized* experience replay** &mdash; Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare or important experience vectors are sampled.
- **Batch Normalization** &mdash; I did not use batch normalization on this project, but I probably should have. I've used batch normalization many times in the past when building convolutional neural networks (CNN), in order to squash pixel values. But, it didn't occur to me that it would be to this project. 


##### &nbsp;

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file.

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Soccer** environment.

![Soccer][image2]

In this environment, the goal is to train a team of agents to play soccer.

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Soccer.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents.  (_To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)





## References

-  <https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf>
-  <https://arxiv.org/pdf/1509.02971.pdf>
- <https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf>
- <https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf>
- <https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf>
- <https://arxiv.org/pdf/1509.02971.pdf>
- <https://machinelearningmastery.com/exploding-gradients-in-neural-networks/>





# Continuous Control

A continuous state space is used to simulate two agents playing tennis.
Eight variables corresponding to the position and velocity of the ball and
racket. There are 33 variables including location, rotation, speed, and angular velocities of each arm.


### Environment

The environment itself has the arm in a continuous space and requires
realistic control. The arm itself is simulating traveling the end of the arm from one point in space to another. No grasping or any other actuation is present.

## Actor-Critic

In this work, the actor-network outputs the action size, which results in an approximation for the value function.

      def forward(self, state):
          """Build an actor (policy) network that maps states -> actions."""
          xs = F.relu(self.fc1(self.bn1(state)))
          x = F.relu(self.fc2(xs))
          return F.tanh(self.fc3(x))

Here we also use the normalization methods to keep any one of the states
from having too large a say in the layer itself. The actor itself is selecting
the next actions

The critic's input is a single state and action. The output is a score of the

      self.fcs1 = nn.Linear(state_size, fcs1_units)
      self.bn1 = nn.BatchNorm1d(state_size)
      self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
      self.fc3 = nn.Linear(fc2_units, 1)

The critic itself is learning about and critiquing the policy the actor comes up with. The critic itself approximates the value function.

## DDPG

The DDPG agent includes facilities for noise, replay buffers, and actor and critic models.



The purpose of adding noise is to perform global exploration rather than local refinement. DDPG's does not explore; therefore, we are using 20 agents with a random noise for each.


## Hyperparameters

      BUFFER_SIZE = int(1e6)  # replay buffer size
      BATCH_SIZE = 1024        # minibatch size
      GAMMA = 0.99            # discount factor
      TAU = 1e-3              # for soft update of target parameters
      LR_ACTOR = 1e-4         # learning rate of the actor
      LR_CRITIC = 1e-3        # learning rate of the critic
      WEIGHT_DECAY = 0        # L2 weight decay



## Future Work

For improving the performance, we may use other approaches such as:

- Reward-Weighted Regression

- Relative Entropy Policy Search

- Recurrent DDPG

