# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:44:36 2019

@author: stu9
"""
# Github zzazeungna~
import numpy as np
import tensorflow as tf

xy = np.loadtxt("c:/data/zoo_data.txt", delimiter=",", usecols=range(1,18), dtype=np.float32)

x_data = xy[:,:-1]
x_data.shape #(101,16)
y_data = xy[:, [-1]]
y_data.shape #(101,1)

x = tf.placeholder(tf.float32, shape=[None,16])
y = tf.placeholder(tf.int32, shape=[None,1])
# y는 one hot encoding을 해주어야 하기 때문에 꼭 int32로 설정해줌

# y_data의 unique한 값의 개수 => 7 (nb_class = 7)
y_one_hot = tf.one_hot(y,7)
y_one_hot = tf.reshape(y_one_hot, [-1,7])

w = tf.Variable(tf.random_normal([16,7],seed=1),name='weight')
b = tf.Variable(tf.random_normal([7],seed=1),name='bias')
''' nb_class = 7
y_one_hot = tf.one_hot(y,nb_class)
y_one_hot = tf.reshape(y_one_hot, [-1,nb_class])

w = tf.Variable(tf.random_normal([16,nb_class],seed=1),name='weight')
b = tf.Variable(tf.random_normal([nb_class],seed=1),name='bias')  '''

logits = tf.matmul(x,w) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_one_hot)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(learning_rate=0.9).minimize(cost)

predict = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(predict, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(train, feed_dict={x:x_data, y:y_data})
    if step % 1000 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={x:x_data, y:y_data})
        print("step{:5}\tloss:{:.3f}\tAcc{:.1%}".format(step, loss, acc))

a = sess.run(hypothesis, feed_dict={x:[[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]]})
print(a, sess.run(tf.argmax(a,1)))
#필요없는 독립변수가 있는경우 제거하고 다시 해보는 것이 좋음
