from __future__ import print_function
from collections import deque

from connect4bot_MCTS import MCTS_Trainer
from connect4game import move

import tensorflow as tf
import numpy as np
import random
import sys
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sess = tf.Session()

# The learning rate for this program decays at each 25% mark through the training
global_step = tf.Variable(0, trainable=False)
minibatches = 400
boundaries = [1, 2, 3]
boundaries = [x * (minibatches//4) for x in boundaries]
values = [0.001, 0.00005, 0.00003, 0.00001]

learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

# Using the learning rate in a standard gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

### Model parameters

# Nodes in the first hidden layer
num_hidden1 = 1024

# Nodes in the second hidden layer
num_hidden2 = 1024

# Board size parameters
num_rows = 6
num_cols = 7

# Batch size for each update
batch_size = 500

# Number of MCTS steps
MCTS_rollouts = 250

# Epsilon-greedy parameter NOTE: Since I'm tracking full MC rollouts, this is not used
epsilon = 0.1

# Mixing parameter
lambda_mix = 0.5


def the_model(data):
	### Define the neural network

	# The input connects to this fully connected layer (6x7x2)[84] -> 1024
	W_fc1 = tf.get_variable("W_fc1", shape=[num_rows*num_cols*2, num_hidden1],
			initializer=tf.contrib.layers.xavier_initializer()
	)
	b_fc1 = tf.get_variable("b_fc1", [num_hidden1],
			initializer=tf.constant_initializer()
	)

	# The previous layer then connects to the next fully connected layer
	W_fc2 = tf.get_variable("W_fc2", shape=[num_hidden1, num_hidden2],
			initializer=tf.contrib.layers.xavier_initializer()
	)
	b_fc2 = tf.get_variable("b_fc2", [num_hidden2],
			initializer=tf.constant_initializer()
	)

	# That second fully connected layer is fully connected to the outputs of the models
	W_fcf = tf.get_variable("W_fcf", shape=[num_hidden2, num_rows*num_cols + 1],
			initializer=tf.contrib.layers.xavier_initializer()
	)
	b_fcf = tf.get_variable("b_fcf", [num_rows*num_cols + 1],
			initializer=tf.constant_initializer()
	)
			
	# Reshape the data and plug it into the network
	flattener = tf.reshape(data, [-1, num_rows*num_cols*2])
	h_fc1 = tf.nn.relu(tf.matmul(flattener, W_fc1) + b_fc1)
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	
	# Finally, return the output of the model, which should be [num_rows*num_cols + 1, 1]
	o = tf.matmul(h_fc2, W_fcf) + b_fcf
	p, v = o[:,0:-1], o[:,-1]
	return p, v


# Restore parameters from a past run?
restore_params = False


# Set up the model
trainer = MCTS_Trainer(sess,
										optimizer,
										the_model,
										restore_params,
										batch_size,
										num_hidden1,
										num_hidden2,
)

saver = tf.train.Saver()


# Function for converting a given board into a string
def concatenate_state_data(Board):
	result = ''
	for row in Board:
		for element in row:
			result += str(element)
	return result


# Functions for updating all N's and Q's for boards that were used
def update_N(N_index, indices, actions):
	for a, i in enumerate(indices):
		N_index[i][actions[a]] = N_index[i][actions[a]] + 1


def update_Q(Q_index, N_index, V_index, lambda_mix, score, indices, actions):
	fun_list = []
	for a, i in enumerate(indices):
		# The indices are lined up such that they're in the same order as moves.
		# Also, Q should be switched between every move because if Player 1 won,
		# then Player 2 must have lost. 
		# So you always start at "score", and then switch score to negative for the next one.
		# If score is 1, then first move is +1, next move is -1, next move is +1, etc.
		# If score is -1, then first move is -1 (lost), next move is +1, next is -1...
		# If score is 0, then score is 0 the entire way through.
		Q_index[i][actions[a]] = Q_index[i][actions[a]] + (1/N_index[i][actions[a]]) * (
				+ (1 - lambda_mix) * V_index[i][actions[a]] 
				+ lambda_mix * score
				- Q_index[i][actions[a]]
		)
		fun_list.append(Q_index[i][actions[a]])
		score = -score


# Function to remove indices not associated with current/future boards in a rollout
def index_cleaner(Board, Board_index, pred_index, P_index, Q_index, N_index, V_index):
	# Convert updated board to its string
	Board_string = concatenate_state_data(Board)

	# Begin a list of now-impossible strings
	unwanted_list = []

	# Check if the pieces in the new board match pieces in the indexed board
	for index, index_string in enumerate(Board_index):
		if len([c for c,d in zip(Board_string, index_string) if c!="0" and c!=d]) > 0:
			unwanted_list.append(index)
			#print(Board_string)
			#print(index_string)

	# Re-make all the lists as only the matching indices
	Board_index = [i for j,i in enumerate(Board_index) if j not in unwanted_list]
	pred_index = [i for j,i in enumerate(pred_index) if j not in unwanted_list]
	P_index = [i for j,i in enumerate(P_index) if j not in unwanted_list]
	Q_index = [i for j,i in enumerate(Q_index) if j not in unwanted_list]
	N_index = [i for j,i in enumerate(N_index) if j not in unwanted_list]
	V_index = [i for j,i in enumerate(V_index) if j not in unwanted_list]

	return Board_index, pred_index, P_index, Q_index, N_index, V_index


# Saved parameters for user output and checking
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


# Constants for deciding actions
Q_coeff = 1
P_coeff = 1
UCB_coeff = 1
dirichlet_factor = 5
dirichlet_mult_constant = 0.5



# Prepare a plot
fig = plt.figure(1)
plt.xlabel("Batch number")
plt.ylabel("Win percentage Versus Random")
red = mpatches.Patch(color='red', label='Wins as Player 1')
blue = mpatches.Patch(color='blue', label='Wins as Player 2')
plt.legend(handles=[red, blue], loc=2)
plt.grid(True)
fig.canvas.draw()
plt.show(block=False)

# A loop over the number of minibatches
for batch_num in range(minibatches):
	#In each step, you do 500 moves, and record the state and action output
	state_list = np.zeros((batch_size, 6, 7, 2), dtype=np.float32)
	action_target_list = np.zeros((batch_size, 42), dtype=np.float32)
	prediction_list = np.zeros((batch_size,), dtype=np.float32)
	outcome_list = np.zeros((batch_size,), dtype=np.float32)
	Game_start = 1

	# A check for whether a test game should be played after a model update
	play_game = 1

	# A loop over the batch size, in each minibatch
	for mini_step in range(batch_size):
		# If it's a new game
		if Game_start == 1:
			# Reset the environment and rewards
			Board = np.zeros((6,7), dtype=int)
			feed_board = np.zeros((1,6,7,2), dtype=np.float32)
			new_feed_board = np.zeros((1,6,7,2), dtype=np.float32)
			player_num = 1
			other_player = 2
			Reward = 0
			action = 0
			start_step = mini_step
			Board_index = []
			pred_index = []
			P_index = []
			N_index = []
			V_index = []
			Q_index = []
			index_list = []
			action_list = []
			Game_start = 0
		

		# The game has been initialized, or is continuing. 
		# Store the board state.
		state_list[mini_step, :, :, :] = feed_board[:]


		# Set the MCTS player numbers
		MCTS_player_num = player_num
		MCTS_other_player = other_player


		# Check to see if the current board has been checked before in this run:
		Board_string = concatenate_state_data(Board)
		if Board_string in Board_index:
			root_index = Board_index.index(Board_string)
		else:
			# Calculate prior probabilities of moves and a win prediction of the current state
			MCTS_P, prediction = trainer.sampleAction(feed_board)
			# Calculate sample values for the following states
			MCTS_V = trainer.sampleValues(feed_board, MCTS_P, MCTS_player_num)
			# Set N and Q to all zeros
			MCTS_N = np.zeros((42,), dtype=int)
			MCTS_Q = np.zeros((42,), dtype=np.float32)

			# Add these values to the indexes
			Board_index.append(Board_string)
			pred_index.append(prediction)
			P_index.append(MCTS_P)
			V_index.append(MCTS_V)
			N_index.append(MCTS_N)
			Q_index.append(MCTS_Q)
			
			# Note down what the index of this board state is
			root_index = len(Board_index) - 1


		# Add in Dirichlet noise to the root P values (NOTE: Lower dirichlet factor -> numbers more spread out)
		D_mask = P_index[root_index] != 0
		P_index[root_index][D_mask] = P_index[root_index][D_mask] + dirichlet_mult_constant * np.random.dirichlet(
				np.ones(np.count_nonzero(P_index[root_index]), dtype=int) * [dirichlet_factor]
		)


		# Store the prediction of how good the current state is
		prediction_list[mini_step] = prediction


		# Some monitoring parameters
		wincheck = 0
		losscheck = 0
		tiecheck = 0
		moves = 0


		for rollout_num in range(MCTS_rollouts):
			#Order of operations:
			#1) Pick a move using P/N/Q/UCB/Dirichlet
			#2) Make that move, get a new MCTS_feed_board
			#3) Use MCTS_feed_board to sample a new action, using P/Q/N, choose greedily
			#4) Do (2,3) until the game is over
			#5) update MCTS_N/Q/P

			# Add the root to the list of all indices to update
			index_list = [root_index]

			# Set MCTS_player_num and MCTS_other_player to the main players
			MCTS_player_num = player_num
			MCTS_other_player = other_player


			# Determine an action, among all legal actions
			pre_action_matrix = (Q_coeff * Q_index[root_index]
					+ P_coeff * P_index[root_index]/(1+N_index[root_index]))
			masked_matrix = np.ma.masked_equal(pre_action_matrix, 0.0, copy=False)
			if sum(N_index[root_index])<2:
				masked_matrix += np.inf
			else:
				masked_matrix += UCB_coeff * np.sqrt(np.log(sum(N_index[root_index]))/N_index[root_index])
			action_index = masked_matrix.argmax()
			action_list = [action_index]
			first_column = (action_index % 7) + 1


			# Perform the move on the starting board, and make that the MCTS board
			MCTS_Board, MCTS_Reward = move(Board, first_column)
			moves += 1


			# Turn the MCTS board into a feed board for the opposing player, for their turn
			MCTS_feed_board = np.zeros((1,6,7,2), dtype=np.float32)
			if not MCTS_Reward:
				for row in range(MCTS_Board.shape[0]):
					for col in range(MCTS_Board.shape[1]):
						if MCTS_Board[row, col] == MCTS_other_player:
							MCTS_feed_board[0, row, col, 0] = 1
						elif MCTS_Board[row, col] != 0:
							MCTS_feed_board[0, row, col, 1] = 1	
				if MCTS_player_num == 1:
					MCTS_other_player = 1
					MCTS_player_num = 2
				else:
					MCTS_player_num = 1
					MCTS_other_player = 2
			elif MCTS_Reward == -1: #Tie game, z_value is 0
				update_N(N_index, index_list, action_list)
				update_Q(Q_index, N_index, V_index, lambda_mix, 0, index_list, action_list)
				tiecheck += 1
			else: #Since this is the first move from the board position, winner must be current player
				update_N(N_index, index_list, action_list)
				update_Q(Q_index, N_index, V_index, lambda_mix, 1, index_list, action_list)
				wincheck += 1


			# Now that the first move has been selected, play the rest of the game using just P/Q/N
			while MCTS_Reward == 0:
				# Check if the current board is already saved in the index
				Board_string = concatenate_state_data(MCTS_Board)
				if Board_string in Board_index:
					index = Board_index.index(Board_string)
				else:
					# Calculate prior probabilities of moves and a win prediction of the current state
					MCTS_P, prediction = trainer.sampleAction(MCTS_feed_board)
					# Calculate smple values for the following states
					MCTS_V = trainer.sampleValues(MCTS_feed_board, MCTS_P, MCTS_player_num)
					# Set N and Q to all zeros
					MCTS_N = np.zeros((42,), dtype=int)
					MCTS_Q = np.zeros((42,), dtype=np.float32)

					# Add these values to the indexes
					Board_index.append(Board_string)
					pred_index.append(prediction)
					P_index.append(MCTS_P)
					V_index.append(MCTS_V)
					N_index.append(MCTS_N)
					Q_index.append(MCTS_Q)
			
					# Note down what the index of this board state is
					index = len(Board_index) - 1


				# Add the current board to the overall index list for later updates
				index_list.append(index)


				# Use the indexed information to find the max of Q(s, a) + u(s, a) in this state
				action_matrix = Q_coeff * Q_index[index] + P_coeff * P_index[index]/(1 + N_index[index])
				masked_matrix = np.ma.masked_equal(action_matrix, 0.0, copy=False)
				action_index = masked_matrix.argmax()
				action_list.append(action_index)
				column = (action_index % 7) + 1


				# Place a piece in that column
				MCTS_Board, MCTS_Reward = move(MCTS_Board, column)


				# Make a new feed board, should be from the perspective of the opposing player
				MCTS_feed_board = np.zeros((1,6,7,2), dtype=np.float32)
				if not MCTS_Reward: #If not, MCTS_feed_board is all zeros
					for row in range(MCTS_Board.shape[0]):
						for col in range(MCTS_Board.shape[1]):
							if MCTS_Board[row, col] == MCTS_other_player:
								MCTS_feed_board[0, row, col, 0] = 1
							elif MCTS_Board[row, col] != 0:
								MCTS_feed_board[0, row, col, 1] = 1
					if MCTS_player_num == 1:
						MCTS_other_player = 1
						MCTS_player_num = 2
					else:
						MCTS_player_num = 1
						MCTS_other_player = 2
				elif MCTS_Reward == -1: #Tie game, z_value is 0
					update_N(N_index, index_list, action_list)
					update_Q(Q_index, N_index, V_index, lambda_mix, 0, index_list, action_list)
					tiecheck += 1
				elif MCTS_player_num == player_num: #If the winner was the current player, z=1
					update_N(N_index, index_list, action_list)
					update_Q(Q_index, N_index, V_index, lambda_mix, 1, index_list, action_list)
					wincheck += 1
				else: #Otherwise the winner is the opposing player, z=-1
					update_N(N_index, index_list, action_list)
					update_Q(Q_index, N_index, V_index, lambda_mix, -1, index_list, action_list)
					losscheck += 1


		# By this point, all of the Monte Carlo rollouts have finished, and a move is chosen. If there is
		# a tie, a move is randomly selected between the tied moves.
		action = np.argmax(np.random.random(N_index[root_index].shape) * (N_index[root_index]==N_index[root_index].max()))


		# Store a normalized vector of the visit counts as the action target for the given board
		action_target_list[mini_step,:] = N_index[root_index] / MCTS_rollouts


		# Perform the move
		Board, Reward = move(Board, (action % 7) + 1)
		print("Visits to each move:")
		print(np.around(np.reshape(N_index[root_index], [6,7]), 3))
		print("Current Board:")
		print(Board)
		print("After {} MC runs, column {} was selected.".format(moves, (action % 7) + 1))
		print("Progress: {} Updates done, {}/{} in the batch".format(batch_num, mini_step+1, batch_size))


		# Translate the new board state into a feed_board for the next move
		feed_board = np.zeros((1,6,7,2), dtype=np.float32)
		if not Reward:
			for row in range(Board.shape[0]):
				for col in range(Board.shape[1]):
					if Board[row, col] == other_player:
						feed_board[0, row, col, 0] = 1
					elif Board[row, col] != 0:
						feed_board[0, row, col, 1] = 1

			if player_num == 1:
				other_player = 1
				player_num = 2
			else:
				player_num = 1
				other_player = 2
		elif Reward == -1:
			outcome_list[start_step:mini_step+1] = [i for i in [0]*(mini_step+1-start_step)]
			Game_start = 1
		else:
			outcome_list[start_step:mini_step+1] = [i for i in [1]*(mini_step+1-start_step)]
			if player_num == 1:
				outcome_list[start_step+1:mini_step:2] = [i for i in [-1]*((mini_step-start_step)//2)]
			else:
				outcome_list[start_step:mini_step:2] = [i for i in [-1]*((mini_step+1-start_step)//2)]
			Game_start = 1
	

	# By this point, we have run all of the moves in the batch. It's time to update the neural network.
	trainer.updateModel(state_list, action_target_list, prediction_list, outcome_list)

	# Save the network.
	save_path = saver.save(sess, "/tmp/simple/model.ckpt")
	print("Model saved in path: %s" % save_path)


	# Play games against a random opponent to check progress
	if play_game:
		go_first = np.zeros([3]) #Format is win/loss/tie
		go_second = np.zeros([3])

		for fake_game in range(1000):
			# Reset the environment and rewards
			Board = np.zeros((6,7), dtype=int)
			feed_board = np.zeros((1,6,7,2), dtype=np.float32)

			# Network is Player 1 for 500 games, then Player 2 for 500 games, vs. random player
			if fake_game < 500:
				network_player = 1
			else:
				network_player = 2

			current_player = 1
			Reward = 0

			while Reward == 0:
				if current_player == network_player:
					actions, _ = trainer.sampleAction(feed_board)
					masked_actions = np.ma.masked_equal(actions, 0.0, copy=False)
					action = masked_actions.argmax()
				else:
					action = np.random.choice(range(7),1)[0]
					while Board[0,action] != 0:
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


	# Print after games against the random player
	if play_game:
		print("Going 1st: {} wins, {} losses, {} ties".format(*go_first))
		print("Going 2nd: {} wins, {} losses, {} ties".format(*go_second))
		game_list.append(batch_num+1)
		first_wins.append(go_first[0]/5)
		second_wins.append(go_second[0]/5)

		one_wins_list.append(one_buffer)
		two_wins_list.append(two_buffer)
		tie_list.append(100-one_buffer-two_buffer)

		# Print to the plot the wins against the random player when the neural network
		# is playing as Player 1 and as Player 2.
		plt.plot(game_list, first_wins, 'r--', game_list, second_wins, 'b--')
		plt.axis([1, batch_num+1, 0, 100])
		fig.canvas.draw()
		time.sleep(0.01)
		sys.stdout.flush()

# This makes sure the ending window doesn't close
plt.show()
