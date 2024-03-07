# DK

```bash
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
```


https://github.com/junxiaosong/AlphaZero_Gomoku



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

Black: MCTS policy trained through the policy value network
White: MCTS policy based solely on pure MCTS

Black win -> Reward: 1
White win -> Reward: -1 or 1e-3 (undecided)
Draw -> Reward: -1


# adjust hyperparameter
### tuning parameter 
n_playout = 20  # = MCTS simulations(n_mcts) & training 2, 20, 50, 100, 400
check_freq = 1  # = more self_playing & training 1, 10, 20, 50, 100


### MCTS parameter
buffer_size = 1000
c_puct = 5
epochs = 10  # During each training iteration, the DNN is trained for 10 epochs.
self_play_sizes = 1
self_play_times = 100 
temperature = 0.1


### Policy update parameter 
batch_size = 64  # previous 512
learn_rate = 2e-4  # previous 2e-3
lr_mul = 1.0
lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
kl_targ = 0.02  # previous 0.02


### Policy evaluate parameter 
win_ratio = 0.0
init_model = None






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
