
# Project 1: Navigation

## Description of the implementation

### Algorithm
In order to solve this challenge, I have explored and implemented the Double Deep Q-Network (DDQN) algorithm.

* [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double Deep Q-Network](https://arxiv.org/abs/1509.06461)

### Approach

In order to solve this challenge, I have implemented the Double Deep Q-Network (DDQN) algorithm. Since DDQN tends to have more stable training, I have chosen it over the Deep Q-Network (DQN) algorithm. 

The algorithm is implemented in the `Navigation.ipynb` notebook. The implementation is based on the [Deep Q-Network (DQN) exercise]. 

I tried multiple model architectures and hyperparameters. The final model architecture is a simple 3-layer fully connected network with 64 and 16 hidden units in each layer as follows: 

```
QNetwork(
  (qnetwork): Sequential(
    (fc0): Linear(in_features=37, out_features=64, bias=True)
    (relu0): ReLU()
    (fc1): Linear(in_features=64, out_features=16, bias=True)
    (relu1): ReLU()
    (fc2): Linear(in_features=16, out_features=4, bias=True)
  )
)
```

The final hyperparameters are:

```
    buffer_size=int(1e5)
    batch_size=128
    gamma=0.99
    tau=1e-3
    lr=8e-4
    lr_decay=0.995
    update_every=4
    n_episodes=700
    eps_start=1.0
    eps_end=0.01
    eps_decay=0.995
```

This model was able to solve the environment in a bit less than 500 episodes, as you can see in the training progress chart below:

![Training Progress](training-progress.png)

> The dashed line represents the target score of 13.0

### Future Work

I would like to explore the following ideas in the future:
1. Implementing the Prioritized Experience Replay.
2. Implementing the Dueling DQN.
3. Trying a few other variants of the DQN algorithm.
