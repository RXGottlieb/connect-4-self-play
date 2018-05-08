# Run using "run_connect4bot_MCTS.py"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import math
from scipy.misc import logsumexp
from connect4game import move


class MCTS_Trainer(object):

	def __init__(self, session,
					optimizer,
					the_model,
					# Argument for restoring parameters
					restore_params,
					# Batch size into the optimizer
					batch_size,
					# Nodes in the first hidden layer
					num_hidden1, # 
					# Nodes in the second hidden layer
					num_hidden2,
					# Board size parameters
					num_rows = 6,
					num_cols = 7,
					# Decay parameter for eligibility trace
					decay = 0.9,
					# Regularization parameter
					reg_param = 0.001,
					# Gradient clipping parameter
					max_gradient = 5):
		
		# TensorFlow machinery
		self.session = session
		self.optimizer = optimizer
		self.restore_params = restore_params

		# Board size
		self.num_rows = num_rows
		self.num_cols = num_cols
		
		# Model components
		self.the_model = the_model

		# Training parameters
		self.num_hidden1 = num_hidden1
		self.num_hidden2 = num_hidden2
		self.batch_size = batch_size
		self.decay = decay
		self.reg_param = reg_param
		self.max_gradient = max_gradient

		# Counter for games played
		self.game_num = 1

		# Saved parameters for user output and checking
		self.actor_P1_loss_tot = 0
		self.actor_P2_loss_tot = 0
		self.critic_loss_tot = 0
		self.total_turns = 0
		self.test = 0
		self.losses = np.array([0,0,0,0,0,0], dtype=float)

		# Initialize the variables
		self.create_variables()

		if not self.restore_params:
			# Create and initialize variables from run_connect4bot_simple.py
			var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			self.session.run(tf.variables_initializer(var_lists))
			self.saver = tf.train.Saver()
			# Make sure the variables are initialized for some reason
			self.session.run(tf.assert_variables_initialized())
		else:
			# If the parameters are to be loaded instead of randomly initialized, load them
			self.saver = tf.train.Saver()
			self.saver.restore(self.session, "/tmp/simple/model.ckpt")
			print("Model restored")


	def create_variables(self):
		# Input board representation as a state
		with tf.name_scope("model_inputs"):
			self.states_for_predict = tf.placeholder(tf.float32, shape=(1, self.num_rows, self.num_cols, 2), name="states")

		# Decide actions within rollouts
		with tf.name_scope("predict_actions"):
			# Initialize the neural networks
			with tf.variable_scope("the_model"):
				self.policy_outputs_P, self.value_output_P = self.the_model(self.states_for_predict)
			
			# Predict an action using the actor model (policy network)
			self.pre_action_scores = tf.identity(self.policy_outputs_P, name="pre_action_scores")

		# Get the lists of model variables
		model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="the_model")

		# Compute the loss using cross entropy and gradients
		with tf.name_scope("compute_pg_gradients"):
			# Accept the list of states
			self.states = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_rows, self.num_cols, 2), name="states")

			# Accept the action targets calculated using MCTS
			self.action_targets = tf.placeholder(tf.float32, shape=(self.batch_size, 42), name="action_targets")

			# Accept the actual outcomes of games
			self.outcomes = tf.placeholder(tf.float32, shape=(self.batch_size,), name="outcomes")

			# Accept the predictions of game outcomes NOTE: I ALREADY RECALCULATE WITH VALUE_OUTPUT
			self.predictions = tf.placeholder(tf.float32, shape=(self.batch_size,), name="predictions")

			# Determine the log probabilities for each action using the actor model
			with tf.variable_scope("the_model", reuse=True):
				self.policy_outputs, self.value_output = self.the_model(self.states)
		
			# Create a mask for valid moves
			self.mask = tf.equal(tf.transpose(tf.one_hot(tf.subtract(
					5*tf.ones([self.batch_size,7], dtype=tf.int64), tf.count_nonzero(tf.reduce_sum(
					self.states, 3), 1)), depth=6), perm = [0, 2, 1]), tf.ones([self.batch_size,6,7], tf.float32)
			)

			# Reshape the mask into a (batch_size, 42) so that I don't have to reshape actions/outputs
			self.flat_mask = tf.reshape(self.mask, shape = [self.batch_size, 42])

			# Apply the mask to the policy outputs
			self.masked_policy_outputs = tf.boolean_mask(self.policy_outputs, self.flat_mask)
			self.masked_action_targets = tf.boolean_mask(self.action_targets, self.flat_mask)

			# Compute the loss as L = (z - v)^2 - cross entropy + regularization*theta
			self.mean_square_loss = tf.reduce_mean(tf.square(self.predictions - self.outcomes))
			self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
					logits = self.masked_policy_outputs,
					labels = self.masked_action_targets
			)
			self.regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in model_variables])
			self.total_loss = self.mean_square_loss + self.cross_entropy_loss + self.reg_param * self.regularization_loss

			# From the loss, calculate the gradients for the model
			self.gradients = self.optimizer.compute_gradients(self.total_loss)

			# Clip the gradients to ensure the model doesn't blow up
			for count, (gradient, variable) in enumerate(self.gradients):
				if gradient is not None:
					self.gradients[count] = (tf.clip_by_norm(gradient, self.max_gradient), variable)

		# Apply the calculated gradients
		with tf.name_scope("train_model"):
			# Use the gradients calculated earlier to update the actor network
			self.train_op = self.optimizer.apply_gradients(self.gradients)
	
	
	def sampleAction(self, state):

		def softmax(y):
			maxy = np.amax(y)
			e = np.exp(y - maxy)
			return e / np.sum(e)

		# Determine action scores by running the model
		action_scores, prediction = self.session.run([
				self.pre_action_scores, 
				self.value_output_P, 
		],{
				self.states_for_predict: state
		})

		# Determine the valid moves, and set the log-likelihood of invalid moves to 0
		valid_moves = np.reshape(np.transpose(np.equal.outer(5 - np.count_nonzero(np.sum(
				state, 3), 1), np.arange(6)).astype(np.int), [0, 2, 1]), [42,]
		)
		action_scores = np.squeeze(action_scores)
		action_scores[valid_moves==0] = float("-inf")

		# Perform a softmax operation on the valid action scores
		valid_scores = softmax(action_scores)

		return valid_scores, prediction

	
	def sampleValues(self, state, actions_to_take, player_num):
		#I'm given the feedboard as state, and the predictions of actions...

		if player_num == 1:
			other_player = 2
		else:
			other_player = 1

		Current_Board = np.zeros((6,7), dtype=int)
		for row in range(state.shape[1]):
			for col in range(state.shape[2]):
				if state[0, row, col, 0] == 1:
					Current_Board[row, col] = player_num
				elif state[0, row, col, 1] == 1:
					Current_Board[row, col] = other_player

		# Now I have the current board, so I need to make a new board for every nonzero action,
		# turn those boards into feed boards, plug those into the neural network, and get values
		# from them.

		prediction_list = np.zeros((42,), dtype=np.float32)
		for i in range(42):
			if actions_to_take[i] != 0:
				New_Board, Reward = move(Current_Board, (i % 7) + 1)

				feed_board = np.zeros((1,6,7,2), dtype=np.float32)
				if not Reward:
					for row in range(New_Board.shape[0]):
						for col in range(New_Board.shape[1]):
							if New_Board[row, col] == other_player:
								feed_board[0, row, col, 0] = 1
							if New_Board[row, col] == player_num:
								feed_board[0, row, col, 1] = 1
				
				prediction = self.session.run([
						self.value_output_P, 
				],{
						self.states_for_predict: feed_board
				})[0]

				prediction_list[i] = -prediction
		
		return prediction_list


	def updateModel(self, states, action_targets, predictions, outcomes):
		# States is [batch_size, 6, 7, 2], of all the different board states
		# Actions is [batch_size, 42], of the action targets for each board
		# Predictions is [batch_size,], of the value estimates for each board
		# Outcomes is [batch_size,], of the actual game outcomes

		# We are given all of the states, targets, predictions, and outcomes.
		# Run a training update.
		_ = self.session.run([
				self.train_op,
		], {
				self.states: states,
				self.action_targets: action_targets,
				self.predictions: predictions,
				self.outcomes: outcomes
		})
