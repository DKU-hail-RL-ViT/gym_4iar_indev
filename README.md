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
pip install mcts
pip install wandb
```

Task : four in a row (9 x 4)


### Problem

그 중에 한가지 에러를 해결함
일단 alphazero의 코드를 따라하고 gym환경에 맞추다보니 몇가지 설정이 덜된듯
alphazero 코드에서는 playout중에 게임이 끝나면 자동으로 root node를 초기화 되는데 
우리 코드에서는 그러지 않았기 때문에 
근데 왜 문제 해결이 안된거 같지




### version  
이건 나중에 다시 정리함

black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

