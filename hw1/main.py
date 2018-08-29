#! py -3
'''
使用以下两种模型回调(1), (2), (3)
1. 抽取全部九小时内的污染源feature的一次项, 加bias
2. 抽取全部九小时内的pm2.5的一次项作为feature, 加bias
备注: 
  a. NR皆为0, 其他不做改动
  b. 所有 advanced 的 gradient descent 技術(如: adam, adagrad 等) 都是可以用的

(1) 記錄誤差值 (RMSE)(根據kaggle public+private分數)，討論兩種feature的影響
(2) 將feature從抽前9小時改成抽前5小時，討論其變化
(3) Regularization on all the weight with λ=0.1、0.01、0.001、0.0001，並作圖(图画出来都差不多)
'''

'''
path: 路径
N: 某一时刻的pm2.5值, 需要前面多少个时段的数值来做输入
features: 需要的features, 默认[], 即全都要
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# 这函数写的有些问题呀， 很慢
def getData(path, N, features=[]):
  data = pd.read_csv(path, encoding='big5').groupby('日期')
  # 再这时就好分好 训练集和验证集了
  # 丢掉无关数据
  res = [vals.drop(['日期','測站'], 1) for name, vals in data]
  if len(features) == 0:
    # 默认的话
    features = res[0].iloc[:, 0].values
  # 过滤无用数据
  res = [val[val['測項'].isin(features)] for val in res]
  # 按时段切分, 将前N个小时的feature变为一行， 最后加上y， 即该时刻的pm2.5值
  data_set = np.array([])
  for val in res:
    # 去除PM2.5的那一行
    pm25 = val[val['測項'] == 'PM2.5']
    for i in range(0, 24 - N):
      t = np.array(val.iloc[:, i + 1:i + 1 + N])
      t = np.append(t, pm25.iloc[0, i + 1 + N]).reshape((1, -1))
      # t即一个数据
      data_set = np.append(data_set, t)
  cols = N * len(features) + 1
  data_set = np.vectorize(lambda num: 0 if num == 'NR' else float(num))(data_set)
  res = np.reshape(data_set, (-1, cols))
  return np.insert(res, 0, 1, axis=1)

def split(data, rate, N):
  # 一天的数据能产生的数据
  count = 24 - N
  start = int(data.shape[0] / count * 0.7) * count
  return data[0: start], data[start:-1]

def generateData(path, name, N, rate, features = []):
  data = getData(path, N, features)
  train, cv = split(data, rate, N)
  pd.DataFrame(train).to_csv('%s_train_.csv' % name)
  pd.DataFrame(cv).to_csv('%s_cv_.csv' % name)

# 获取测试的
def getTestData(path, N, features = []):
  data = pd.read_csv(path, encoding='big5').groupby('id_0')
  data = [val for name, val in data]
  if len(features) == 0:
    # 默认的话
    features = data[0].iloc[:, 1].values
  data = [val[val['AMB_TEMP'].isin(features)] for val in data]
  data = [np.array(val.iloc[:, 11-N:]).reshape(1, -1) for val in data]
  data = np.insert(data, 0, 1, axis=1)
  return data

def initData():
  generateData('train.csv', 'all_9', 9, 0.7)
  generateData('train.csv', 'pm25_9', 9, 0.7, ['PM2.5'])
  generateData('train.csv', 'all_5', 5, 0.7)
  generateData('train.csv', 'pm25_5', 5, 0.7, ['PM2.5'])

def loss(x, y, theta, _lambda):
  res = np.sum((y - np.dot(x, theta))**2) + _lambda * np.sum(theta**2)
  res = res / (2 * x.shape[0])
  return res


def gradientDescent(data, rate, _lambda):
  cols = data.shape[1] - 1
  row = data.shape[0]
  # theta的初始值要为1呀， 为0的我是再想什么呀
  theta = np.ones(cols)
  gt = np.zeros(cols)
  np.random.shuffle(data)
  repeat = 0
  times = 100
  losses = np.zeros(times)
  X = data[:, 0:-1]
  Y = data[:, -1]
  while repeat < times:
    g = (Y - X @ theta) @ (-X) + _lambda * theta
    g = g / row
    gt = gt + g**2
    theta = theta - rate * g / (gt**0.5)
    losses[repeat] = loss(X, Y, theta, _lambda)
    repeat += 1
  print(loss(X, Y, theta, _lambda))
  axiosX = np.linspace(0, repeat - 1, repeat)
  return axiosX, losses, theta

def a3():
  data = np.array(pd.read_csv('./all_9_train_.csv'))
  data = data[:, 1:]
  cv_data = np.array(pd.read_csv('./all_9_cv_.csv'))
  cv_data = cv_data[:, 1:]
  CX = cv_data[:, 0:-1]
  CY = cv_data[:, -1]
  # x1, y1, theta1 = gradientDescent(data, 0.1, 0.1)
  # x2, y2, theta2 = gradientDescent(data, 0.08, 0.1)
  # x3, y3, theta3 = gradientDescent(data, 0.06, 0.1)
  # x4, y4, theta4 = gradientDescent(data, 0.2, 0.01)

  # plt.plot(x1, y1, label="0.1")
  # plt.plot(x1, y2, label="0.01")
  # plt.plot(x1, y3, label="0.001")
  # plt.plot(x4, y4, label="0.0001")
  

  rates = [0.1, 0.01, 0.001]
  lambdas = [0.1, 0.01, 0.001, 0.0001]
  for rate in rates:
    for _lambda in lambdas:
      x, y, theta = gradientDescent(data, rate, _lambda)
      text = "rate=%s, lambda=%s" % (rate, _lambda)
      plt.plot(x, y, label=text)
      print(text)
      print(loss(CX, CY, theta, _lambda))

  plt.xlabel('update time')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

def a1(train, cv, test):
  train_data = np.array(pd.read_csv(train))
  train_data = train_data[:, 1:]
  cv_data = np.array(pd.read_csv(cv))
  cv_data = cv_data[:, 1:]
  x1, y1, theta1 = gradientDescent(train_data, 0.00001, 0.1)

def a4():
  data = np.array(pd.read_csv('./all_9_train_.csv'))
  data = data[:, 1:]
  cv_data = np.array(pd.read_csv('./all_9_cv_.csv'))
  cv_data = cv_data[:, 1:]
  X = data[:, 0:-1]
  Y = data[:, -1]
  CX = cv_data[:, 0:-1]
  CY = cv_data[:, -1]
  theta = getCorrectTheta(X, Y)
  print(loss(CX, CY, theta, 0.1))

def getCorrectTheta(X, Y):
  theta = np.linalg.inv(X.T @ X) @ X.T @ Y
  l = loss(X, Y, theta, 0.1)
  print(l)
  return theta

def main():
  #1 读取
  # test = getTestData('./test.csv', 9)
  # print(test.shape)
  #2 生成所需数据
  #3 分成train vc 集, 7:3
  #4 开始训练
  # a4()

  a3()

if __name__ == '__main__':
  main()