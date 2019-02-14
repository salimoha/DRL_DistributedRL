# Train Two RL Agents to Play Tennis
##### &nbsp;

## Background
For artificial intelligence (AI) to reach its full potential, AI systems need to interact safely and efficiently with humans, as well as other agents. There are already environments where this happens on a daily basis, such as the stock market. And there are future applications that will rely on productive agent-human interactions, such as self-driving cars and other autonomous vehicles.

One step along this path is to train AI agents to interact with other agents in both cooperative and competitive settings. Reinforcement learning (RL) is a subfield of AI that's shown promise. However, thus far, much of RL's success has been in single agent domains, where building models that predict the behavior of other actors is unnecessary. As a result, traditional RL approaches (such as Q-Learning) are not well-suited for the complexity that accompanies environments where multiple agents are continuously interacting and evolving their policies.

[//]: # (> Unfortunately, traditional reinforcement learning approaches such as Q-Learning or policy gradient
are poorly suited to multi-agent environments. One issue is that each agent’s policy is changing
as training progresses, and the environment becomes non-stationary from the perspective of any
individual agent in a way that is not explainable by changes in the agent’s own policy. This presents
learning stability challenges and prevents the straightforward use of past experience replay, which is crucial for stabilizing deep Q-learning. Policy gradient methods, on the other hand, usually exhibit very high variance when coordination of multiple agents is required. Alternatively, one can use model-based policy optimization which can learn optimal policies via back-propagation, but this requires
a differentiable model of the world dynamics and assumptions about the interactions between
agents. Applying these methods to competitive environments is also challenging from an optimization
perspective, as evidenced by the notorious instability of adversarial training methods [11].)

[//]: # (https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)

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

