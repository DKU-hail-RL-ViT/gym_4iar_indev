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

mcts.py 143번째 줄 _evaluate_rollout 메서드 부터 수정해야함
잘 돌아가는지는 잘 모르겠는데 일단 _playout까지는 코드상 돌아가는건 문제 없는거 같음

그다음으로 해야할 건mcts 할때 player나눠주는거 
start self play 메서드 (190번째 줄)
p1, p2 = self.board.players
이부분 처럼 player를 나눠주고
돌을 두고 하는 식으로



간헐적으로 obs[3] 이 36이 아닌데 draw가 찍힘
이게 왜 그러는건지 잘 모르겠음
draw일 떄도 간혹 이러는 거라 디버깅 찍어보기도 쉽지 않음




### version

black 학습, white 랜덤

black win -> reward : 1
white win -> reward : abs(-1)
draw -> reward : 0

player_myself = 0 , player_enemy = 1 일때 흑이 돌을 놓을 차례
player_myself = 1 , player_enemy = 0 일때 백이 돌을 놓을 차례

