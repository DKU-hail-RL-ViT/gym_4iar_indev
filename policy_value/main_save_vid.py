#
import torch
import os
from matplotlib import pyplot as plt
file_name = '~~~.pth' #player 1
file_name2 = '~~~~.pth' #player 2

save_dir = './vid/'
os.makedirs(os.path.join(save_dir, file_name.split('.')[0]), exist_ok=True)

# load file_name
model = torch.load(file_name)#?


fig, ax = plt.subplots()
# 0 ~ 3 0 ~8

for ep in range(1):
	# initialize env
	ax.set_xlim(0, 4)
	ax.set_ylim(0, 9)
	ax.plot(0+0.5, 0+0.5, 'o', color='black', markersize=20)
	# https://ransakaravihara.medium.com/how-to-create-gifs-using-matplotlib-891989d0d5ea
	# https://stackoverflow.com/questions/25140952/matplotlib-save-animation-in-gif-error
	# https://pinkwink.kr/860


