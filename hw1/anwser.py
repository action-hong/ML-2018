#! py -3
import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []

for i in range(18):
  data.append([])

n_row = 0
text = open('./train.csv', 'r', encoding='big5')
# 行
row = csv.reader(text, delimiter=',')
for r in row:
  if n_row != 0:
    # 每一行只有3-27格有
    for i in range(3, 27):
      val = float(0)
      if r[i] != 'NR':
        val = r[i]
      # 关键呀这里, 难怪我之前做的误差那么大,
      # 前一天的数据也可以作为今天数据的输入呀
      # 所以这里做一个 18 * N 的数组， 擦擦
      data[(n_row - 1) % 18].append(val)
  n_row = n_row + 1
text.close()

x = []
y = []
# 12个月
for i in range(12):
  # 一个月连续10小时的data有471笔(其实这个不大准确呀)
  # 为毛是471？
  # 因为训练数据中， 取的是每个月的前20天， 故有480小时， 所以是471
  for j in range(471):
    x.append([])
    for t in range(18):
      for s in range(9):
        x[471*i+j].append(data[t][480*i+j+s])
    y.append(data[9][480 * i + j + 9])
x = np.array(x)
x = np.asarray(x, dtype=float)
y = np.array(y)
y = np.asarray(y, dtype=float)

# add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000

w_r = np.linalg.inv(x.T @ x) @ x.T @ y

x_t = x.T
s_gra = np.zeros(len(x[0]))
print(x.dtype)
print(w.dtype)
# for i in range(repeat):
#   hypo = np.dot(x,w)
#   loss = hypo - y
#   cost = np.sum(loss**2) / len(x)
#   cost_a  = math.sqrt(cost)
#   gra = x_t @ loss
#   s_gra += gra**2
#   ada = np.sqrt(s_gra)
#   w = w - l_rate * gra/ada
#   print('iteration: %d | Cost: %f ' % (i, cost_a))


# # save model
# np.save('./model.npy', w)

# read model
w = np.load('./model.npy')

# 正确答案
# hypo = np.dot(x,w_r)
# loss = hypo - y
# cost = np.sum(loss**2) / len(x)
# cost_a  = math.sqrt(cost)
# print('correct: %d | Cost: %f ' % (i, cost_a))

# test
test_x = []
n_row = 0
text = open('./test.csv', 'r')
row = csv.reader(text, delimiter=',')

for r in row:
  # 即一个新的日期
  if n_row % 18 == 0:
    test_x.append([])
    for i in range(2, 11):
      test_x[n_row // 18].append(float(r[i]))
  else:
    for i in range(2, 11):
      val = float(0)
      if r[i] != 'NR':
        val = float(r[i])
      test_x[n_row // 18].append(val)
  n_row = n_row + 1
text.close()
test_x = np.array(test_x)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# ans = []
# for i in range(len(test_x)):
#   ans.append(["id_"+str(i)])
#   a = np.dot(w,test_x[i])
#   ans[i].append(a)

# filename = "result/predict.csv"
# text = open(filename, "w+")
# s = csv.writer(text,delimiter=',',lineterminator='\n')
# s.writerow(["id","value"])
# for i in range(len(ans)):
#     s.writerow(ans[i]) 
# text.close()

# read answer
text = open('./ans.csv', 'r')
row = csv.reader(text, delimiter=',')
n_row = 0
ans_y = []
for r in row:
  if n_row != 0:
    ans_y.append(r[1])
  n_row = n_row + 1
ans_y = np.array(ans_y)
text.close()


hypo = np.dot(test_x, w)
hypo = np.asarray(hypo, dtype=float)
ans_y = np.asarray(ans_y, dtype=float)
loss = hypo - ans_y
cost = np.sum(loss**2) / len(test_x)
cost_a  = math.sqrt(cost)
print('test | Cost: %f ' % (cost_a))
