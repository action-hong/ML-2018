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
(3) Regularization on all the weight with λ=0.1、0.01、0.001、0.0001，並作圖
'''

'''
path: 路径
N: 某一时刻的pm2.5值, 需要前面多少个时段的数值来做输入
features: 需要的features, 默认[], 即全都要
'''

import pandas as pd

def getData(path, N, features=[]):
  data = pd.read_csv(path, encoding='big5').groupby('日期')
  # 丢掉无关数据
  res = [vals.drop(['日期','測站'], 1) for name, vals in data]
  res = [vals for i in range(0, 24 - N) for index, vals in res]

def main():
  #1 读取
  N = 9

  #2 生成所需数据
  #3 分成train vc 集, 7:3
  #4 开始训练

if __name__ == '__main__':
  main()