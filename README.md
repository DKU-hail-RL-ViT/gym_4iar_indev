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
pip install sb3-contrib
pip install 'stable-baselines3[extra]'
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

mcts 111번째 줄
playout 하는 부분에서 문제가 생기는 듯




### version  
이건 나중에 다시 정리함

black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

