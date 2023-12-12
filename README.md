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

mcts_pure 160 ㄱㄱ


추가해야할건 루트 노드가 뭔지 나오게한다면 좋을거 같다고 하심

질문1 mcts  winning에서 리턴 되는 값에 따라 누구를 학습시킬지가 결정이 되는데

질문2 mcts_pure _evaluate_rollout 163번줄
얘는 흰돌을 기준으로 하니까 return값을 반대로 줘야 백의 승률을 올려주는건가





### version  

교수님이 해결해주신 version
batch_size = 128   # previous 512 너무 오래걸려서 128로 줄여놓았음





black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

