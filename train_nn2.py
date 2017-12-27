import tensorflow as tf
import numpy as np
#from processdata import create_feature_sets_and_labels
from my_data_parse import bring_processed, get_train_data_batch, get_test_data
print("Started loading data")
# train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
#train_x,train_y,test_x,test_y = bring_processed()

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 100

n_classes = 2
batch_size = 100
hm_epochs = 10

train_data_batch_gen = get_train_data_batch(batch_size)
train_x,train_y = train_data_batch_gen.__next__()
print("intitial training load done")
test_x,test_y  = get_test_data()
print("loading done")

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float', [None, len(train_y[0])])

def preformance_metrix(pred, test):
	tp=0
	tn=0
	fp=0
	fn=0
	for i,j in zip(pred, test):
	    if i==1 and j == 1:
	        tp+=1
	    if i==0 and j == 0:
	        tn+=1
	    if i==1 and j == 0:
	        fp+=1
	    if i==0 and j==1:
	        fn+=1
	#print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp, tn, fp, fn))

	print("sensitivity: ", tp/(tp+fn))

	print("specificity: ", tn/(tn+fp))

	print("accuracy: ", (tp+tn)/(tp+fp+fn+tn))


def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	# hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	# 				  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	#
	# hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	# 				  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes])),}


	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	# l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	# l2 = tf.nn.relu(l2)
	#
	# l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	# l3 = tf.nn.relu(l3)

	output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


	hm_epochs = 2

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			train_data_batch_gen = get_train_data_batch(batch_size)
			epoch_loss = 0
			i=0
			while i < 24000:
				start = i
				end = i+batch_size
				train_x, train_y = train_data_batch_gen.__next__()


				batch_x = np.array(train_x)
				batch_y = np.array(train_y)

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
															  y: batch_y})
				epoch_loss += c
				i+=batch_size



			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			#print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))
			pred = tf.argmax(prediction, 1)
			test = tf.argmax(y, 1)

			pred_val = pred.eval({x:test_x})
			test_val = test.eval({y:test_y})

			pred_lst = pred_val.tolist()
			test_lst = test_val.tolist()

			print("epoch {}".format(epoch))
			preformance_metrix(pred_lst, test_lst)

		print("____final______")
		pred = tf.argmax(prediction, 1)
		test = tf.argmax(y, 1)

		pred_val = pred.eval({x:test_x})
		test_val = test.eval({y:test_y})

		pred_lst = pred_val.tolist()
		test_lst = test_val.tolist()

		preformance_metrix(pred_lst, test_lst)





train_neural_network(x)
