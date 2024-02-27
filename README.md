# DK

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
pip install torch
pip install gymnasium
pip install scipy
pip install pyglet
```

# SH

python version 3.10

```bash
pip install torch
pip install gymnasium
pip install pyglet
pip install scipy==1.11.1
pip install numpy==1.25.1
pip install matplotlib==3.7.2
pip install gym==0.26.2
pip install pyglet==2.0.8
pip install wandb
```

Task : four in a row (9 x 4)


### Problem






### version  
수정해야함

Black: MCTS policy trained through the policy value network
White: MCTS policy based solely on pure MCTS

Black win -> Reward: 1
White win -> Reward: -1 or 1e-3 (undecided)
Draw -> Reward: -1


## adjust hyperparameter :  policy_value.train_fiar.py 285
batch_size = 128   # previous 512  (toooo slow)


## adjust hyperparameter :  policy_value.train_fiar.py 13 ~ 33
n_playout = 200  # previous 400
pure_mcts_playout_num = 500     # previous 1000 



self_play_sizes = 1
temp = 1e-3
buffer_size = 10000
epochs = 5  # num of train_steps for each update
self_play_times = 1000   # previous 1500
pure_mcts_playout_num = 500     # previous 1000







## Summary

the overall process can be broadly divided into the self-play and start-play phases. 

During self-play, a single Monte Carlo Tree Search (MCTS) alternately plays as both black and white, 
learning through a policy-value network composed of a Multi-Layer Perceptron (MLP), 
akin to an actor-critic model.

The MCTS trained during self-play becomes MCTS 1, and in subsequent games, MCTS 2 is set to choose actions 
purely through MCTS without passing through the policy-value network. 

In the start-play phase, MCTS 1 and MCTS 2, now representing black and white policies, engage in games. 
By default, MCTS 1 is configured to play as black.

The win rate of MCTS 1 during these games is used as a benchmark, and the corresponding policy version 
is saved if the win rate is higher. 
This saved policy version is then compared with the ongoing policies, allowing for the continuous refinement 
of policies based on their performance. 
