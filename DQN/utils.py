import gym
import matplotlib.pyplot as plt
import numpy as np

def preprocess(image,constant):
  '''
  (210, 160, 3)->(160, 160)->(80, 80)
  '''
  image = image[34:194,:,:]              # 裁剪
  image = np.mean(image,axis=2,keepdims=False)    # 灰度处理
  image = image[::2,::2]                # 下采样
  # 归一化操作有：固定mean和std，根据min和max缩放，归一化
  image = image/256                   # 归一化
  image = image - constant/256            # 根据背景值去除背景为0
  return image

def showplot():
  rewards = np.load('Pong-v0_rewards.npy')
  plt.plot(rewards)
  plt.show()

  average = [np.mean(rewards[i-100:i]) for i in range(100,len(rewards))]
  plt.plot(average)
  plt.show()