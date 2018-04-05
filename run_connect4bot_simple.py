from __future__ import print_function
from collections import deque

from connect4bot_simple import ActionValueActorCritic
from connect4game import move

import tensorflow as tf
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sess = tf.Session()

critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
actor_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#state_dim = [1,6,7,2]
#num_actions = [6,7]

# Nodes in the first hidden layer
num_hidden1 = 1024

# Nodes in the second hidden layer
num_hidden2 = 128

# Board size parameters
num_rows = 6
num_cols = 7

# Batch size into the optimizer
batch_size = 1


def actor_model_P1(data):
	### Define the policy neural network for player 1

	# The input connects to this fully connected layer (6x7x2)[84] -> 1024
	W_fc1 = tf.get_variable("W_fc1", shape=[num_rows*num_cols*2, num_hidden1],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.get_variable("b_fc1", [num_hidden1],
			initializer=tf.constant_initializer())

	# The previous layer then connects to this smaller fully connected layer
	W_fc2 = tf.get_variable("W_fc2", shape=[num_hidden1, num_hidden2],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable("b_fc2", [num_hidden2],
			initializer=tf.constant_initializer())

	# That smaller fully connected layer is fully connected to the outputs of the models
	W_fcf = tf.get_variable("W_fcf", shape=[num_hidden2, num_rows*num_cols],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fcf = tf.get_variable("b_fcf", [num_rows*num_cols],
			initializer=tf.constant_initializer())
			
	# Reshape the data and plug it into the network, with dropout if this is during training
	flattener = tf.reshape(data, [-1, num_rows*num_cols*2])
	h_fc1 = tf.nn.relu(tf.matmul(flattener, W_fc1) + b_fc1)
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
	# Finally, return the output of the model, which should be [num_rows*num_cols, 1]
	p = tf.matmul(h_fc2, W_fcf) + b_fcf
	return p

def actor_model_P2(data):
	### Define the policy neural network for player 2

	# The input connects to this fully connected layer (6x7x2)[84] -> 1024
	W_fc1 = tf.get_variable("W_fc1", shape=[num_rows*num_cols*2, num_hidden1],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.get_variable("b_fc1", [num_hidden1],
			initializer=tf.constant_initializer())

	# The previous layer then connects to this smaller fully connected layer
	W_fc2 = tf.get_variable("W_fc2", shape=[num_hidden1, num_hidden2],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable("b_fc2", [num_hidden2],
			initializer=tf.constant_initializer())

	# That smaller fully connected layer is fully connected to the outputs of the models
	W_fcf = tf.get_variable("W_fcf", shape=[num_hidden2, num_rows*num_cols],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fcf = tf.get_variable("b_fcf", [num_rows*num_cols],
			initializer=tf.constant_initializer())
			
	# Reshape the data and plug it into the network, with dropout if this is during training
	flattener = tf.reshape(data, [-1, num_rows*num_cols*2])
	h_fc1 = tf.nn.relu(tf.matmul(flattener, W_fc1) + b_fc1)
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
	# Finally, return the output of the model, which should be [num_rows*num_cols, 1]
	p = tf.matmul(h_fc2, W_fcf) + b_fcf
	return p

def critic_model(data):
	### Define the critic neural network (same architecture as actor network but with scalar output)

	# The input connects to this fully connected layer
	W_fc1 = tf.get_variable("W_fc1", shape=[num_rows*num_cols*2, num_hidden1],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.get_variable("b_fc1", [num_hidden1],
			initializer=tf.constant_initializer())

	# The previous layer then connects to this smaller fully connected layer
	W_fc2 = tf.get_variable("W_fc2", shape=[num_hidden1, num_hidden2],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable("b_fc2", [num_hidden2],
			initializer=tf.constant_initializer())

	# That smaller fully connected layer is fully connected to the outputs of the models
	W_fcf = tf.get_variable("W_fcf", shape=[num_hidden2, 1],
			initializer=tf.contrib.layers.xavier_initializer())
	b_fcf = tf.get_variable("b_fcf", [1],
			initializer=tf.constant_initializer())
			
	# Reshape the data and plug it into the network, with dropout if this is during training
	flattener = tf.reshape(data, [-1, num_rows*num_cols*2])
	h_fc1 = tf.nn.relu(tf.matmul(flattener, W_fc1) + b_fc1)
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
	# Finally, return the output of the model, which should be scalar
	v = tf.matmul(h_fc2, W_fcf) + b_fcf
	return v

pg_actorcritic = ActionValueActorCritic(sess,
										actor_optimizer,
										critic_optimizer,
										actor_model_P1,
										actor_model_P2,
										critic_model)

saver = tf.train.Saver()

MAX_GAMES = 10001
one_wins = 0
two_wins = 0
old_one = 0
old_two = 0

game_list = []
first_wins = []
second_wins = []
one_buffer = 0
two_buffer = 0
one_wins_list = []
two_wins_list = []
tie_list = []

# Now that the models are all set, and the variables are initialized (in ActionValueActorCritic),
# we can start with the actual training of the model.

show_game = 0

fig = plt.figure(1)

plt.subplot(211)
plt.xlabel("Game number")
plt.ylabel("Win percentage Versus Random")
red = mpatches.Patch(color='red', label='Wins as Player 1')
blue = mpatches.Patch(color='blue', label='Wins as Player 2')
plt.legend(handles=[red, blue], loc=2)
plt.grid(True)
plt.subplot(212)
plt.xlabel("Game number")
plt.ylabel("Win percentage During Training")
red = mpatches.Patch(color='red', label='Player 1 Wins')
blue = mpatches.Patch(color='blue', label='Player 2 Wins')
green = mpatches.Patch(color='green', label='Tie Games')
plt.legend(handles=[red, blue, green], loc=2)
fig.canvas.draw()
plt.show(block=False)
		
for game_num in range(1, MAX_GAMES+1):

	# Print the game if desired
	if (game_num % 500)-1 == 0:
		play_game = 1
	else:
		play_game = 0

	if (game_num % 5000)-1 == 0:
		show_game = 1
	else:
		show_game = 0

	# Reset the environment and rewards
	Board = np.zeros((6,7), dtype=int)
	feed_board = np.zeros((1,6,7,2), dtype=np.float32)
	new_feed_board = np.zeros((1,6,7,2), dtype=np.float32)
	player_num = 1
	other_player = 2
	Reward = 0
	column = 0

	while Reward == 0:
		# Decide on an action
		action = pg_actorcritic.sampleAction(feed_board, player_num)

		# Translate action into a column to move in
		column = (action % 7) + 1

		# Make that action
		New_Board, Reward = move(Board, column)

		# Reformat the next board into [1 x 6 x 7 x 2]
		# Should be from the perspective of NOT player_num
		new_feed_board = np.zeros((1,6,7,2), dtype=np.float32)
		if Reward == 0: #If not, new_feed_board is all zeros
			for row in range(New_Board.shape[0]):
				for col in range(New_Board.shape[1]):
					if New_Board[row, col] == other_player:
						new_feed_board[0, row, col, 0] = 1
					elif New_Board[row, col] != 0:
						new_feed_board[0, row, col, 1] = 1

		if player_num == 1:
			pg_actorcritic.updateP1Model(feed_board, action, Reward, new_feed_board, show_game)
		else:
			pg_actorcritic.updateP2Model(feed_board, action, Reward, new_feed_board, show_game)

		# Update the board, and update the feedboard
		Board = New_Board

		"""if show_game:
			print("Action chosen: {}".format(column))
			print(Board)
			if Reward == 0:
				print("Player #{0} (Game #{1})".format(player_num, game_num))

			elif Reward == 1:
				print("Player #{0}, (Game #{1}) WINNER".format(player_num, game_num))

			else:
				print("Player #{0}, (Game #{1}) TIE GAME".format(player_num, game_num))"""

		# Swap player's turn
		if not Reward:
			if player_num == 1:
				other_player = 1
				player_num = 2
			else:
				player_num = 1
				other_player = 2

		feed_board = np.zeros((1,6,7,2), dtype=np.float32)
		for row in range(Board.shape[0]):
			for col in range(Board.shape[1]):
				if Board[row, col] == player_num:
					feed_board[0, row, col, 0] = 1
				elif Board[row, col] != 0:
					feed_board[0, row, col, 1] = 1			
	if Reward == 1:
		if player_num == 1:
			one_wins += 1
		else:
			two_wins += 1

	# This point is reached once a game is finished

	if play_game:
		go_first = np.zeros([3]) #Format is win/loss/tie
		go_second = np.zeros([3])

		for fake_game in range(1000):
			# Reset the environment and rewards
			Board = np.zeros((6,7), dtype=int)
			feed_board = np.zeros((1,6,7,2), dtype=np.float32)

			# Network is player 1 500 games, player 2 500 games, vs. random player
			if fake_game < 500:
				network_player = 1
			else:
				network_player = 2

			current_player = 1
			Reward = 0

			while Reward == 0:
				if current_player == network_player:
					if network_player == 1:
						action = pg_actorcritic.sampleAction(feed_board, 1)
					if network_player == 2:
						action = pg_actorcritic.sampleAction(feed_board, 2)
				else:
					action = np.random.choice(range(7),1)[0]

				column = (action % 7) + 1
				
				Board, Reward = move(Board, column)

				# Swap player's turn
				if not Reward:
					if current_player == 1:
						current_player = 2
					else:
						current_player = 1

				# Reset, then reformat the current board into [1 x 6 x 7 x 2] for the next move (if applicable)
				feed_board = np.zeros((1,6,7,2), dtype=np.float32)
				for row in range(Board.shape[0]):
					for col in range(Board.shape[1]):
						if Board[row, col] == current_player:
							feed_board[0, row, col, 0] = 1
						elif Board[row, col] != 0:
							feed_board[0, row, col, 1] = 1

			if Reward == -1: #Indicates a tie
				if network_player == 1:
					go_first[2] += 1
				else:
					go_second[2] += 1
			elif current_player == network_player: #then the reward must be 1, and network player must have won
				if network_player == 1:
					go_first[0] += 1
				else:
					go_second[0] += 1
			else: #A player won, but it was not the network_player
				if network_player == 1:
					go_first[1] += 1
				else:
					go_second[1] += 1

	# Print every 100 games
	if game_num % 100 == 0:
		print("Total training games: {}".format(game_num))
		print("Wins as Player 1: {} ({})".format(one_wins, one_wins - old_one))
		print("Wins as Player 2: {} ({})".format(two_wins, two_wins - old_two))
		one_buffer = one_wins - old_one
		two_buffer = two_wins - old_two
		old_one = one_wins
		old_two = two_wins

	# Print after games against the random player, and save the model
	if play_game:
		print("Going 1st: {} wins, {} losses, {} ties".format(*go_first))
		print("Going 2nd: {} wins, {} losses, {} ties".format(*go_second))
		save_path = saver.save(sess, "/tmp/simple/model.ckpt")
		print("Model saved in path: %s" % save_path)
		game_list.append(game_num)
		first_wins.append(go_first[0]/5)
		second_wins.append(go_second[0]/5)

		one_wins_list.append(one_buffer)
		two_wins_list.append(two_buffer)
		tie_list.append(100-one_buffer-two_buffer)

		plt.subplot(211)
		plt.plot(game_list, first_wins, 'r--', game_list, second_wins, 'b--')
		plt.axis([0, game_num, 0, 100])

		plt.subplot(212)
		plt.plot(game_list, one_wins_list, 'r--', game_list, two_wins_list, 'b--', game_list, tie_list, 'g--')
		plt.axis([0, game_num, 0, 100])

		fig.canvas.draw()


		time.sleep(0.01)
		
		sys.stdout.flush()

# This makes sure the ending window doesn't close
plt.show() 
