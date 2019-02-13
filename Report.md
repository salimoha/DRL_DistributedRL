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

