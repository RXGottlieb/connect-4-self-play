#  The python version of Connect 4, using TensorFlow.

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


class ActionValueActorCritic(object):

	def __init__(self, session,
					actor_optimizer,
					critic_optimizer,
					actor_model_P1,
					actor_model_P2,
					critic_model,
					# Nodes in the first hidden layer
					num_hidden1 = 1024, # 
					# Nodes in the second hidden layer
					num_hidden2 = 128,
					# Board size parameters
					num_rows = 6,
					num_cols = 7,
					# Batch size into the optimizer
					batch_size = 1,
					# Decay parameter for eligibility trace
					decay = 0.9,
					# Regularization parameter
					reg_param = 0.001,
					# Gradient clipping parameter
					max_gradient = 5):
		
		# TensorFlow machinery
		self.session = session
		self.actor_optimizer = actor_optimizer
		self.critic_optimizer = critic_optimizer
		self.RESTORE_HERE = False

		# Board size
		self.num_rows = num_rows
		self.num_cols = num_cols
		
		# Model components
		self.actor_model_P1 = actor_model_P1
		self.actor_model_P2 = actor_model_P2
		self.critic_model = critic_model

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

		if not self.RESTORE_HERE:
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
			self.state = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_rows, self.num_cols, 2))

		# Decide actions within rollouts
		with tf.name_scope("predict_actions"):
			# Initialize the neural networks
			with tf.variable_scope("actor_model_P1"):
				self.P1_policy_outputs = self.actor_model_P1(self.state)

			with tf.variable_scope("actor_model_P2"):
				self.P2_policy_outputs = self.actor_model_P2(self.state)

			with tf.variable_scope("critic_model"):
				self.value_output = self.critic_model(self.state)
			
			# Predict an action using the actor model (policy network)
			self.P1_pre_action_scores = tf.identity(self.P1_policy_outputs, name="P1_pre_action_scores")
			self.P2_pre_action_scores = tf.identity(self.P2_policy_outputs, name="P2_pre_action_scores")

		# Get the lists of model variables
		actor_model_P1_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_model_P1")
		actor_model_P2_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_model_P2")
		critic_model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_model")

		# Compute the loss using cross entropy and gradients
		with tf.name_scope("compute_pg_gradients"):
			# Accept as input the next state
			self.next_state = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_rows, self.num_cols, 2))

			# Accept as input the action that was taken NOTE: This is now a 42-long guy
			self.action_taken = tf.placeholder(tf.int32, (None,), name="action_taken")

			# Accept as input the discounted reward for this step
			self.reward = tf.placeholder(tf.float32, (None,), name="reward")

			# Determine the log probabilities for each action using the actor model
			with tf.variable_scope("actor_model_P1", reuse=True):
				self.P1_actor_output = self.actor_model_P1(self.state)
				#self.log_prob_pre = self.actor_model(self.state)
				#self.log_probabilities = tf.nn.softmax(self.actor_model(self.state))
			
			with tf.variable_scope("actor_model_P2", reuse=True):
				self.P2_actor_output = self.actor_model_P2(self.state)
			
			# Determine the estimated value of the state (exponential to be between 0 and 1)
			with tf.variable_scope("critic_model", reuse=True):
				self.estimated_value = self.critic_model(self.state)
				self.next_value = tf.cond(tf.equal(tf.count_nonzero(self.next_state), tf.zeros([], dtype=tf.int64)), 
						lambda: tf.zeros([], dtype=tf.float32),
						lambda: 1-self.critic_model(self.next_state)
				)
	
			# Create a mask for valid moves
			self.mask = tf.equal(tf.transpose(tf.one_hot(tf.subtract(
					5*tf.ones([1,7], dtype=tf.int64), tf.count_nonzero(tf.reduce_sum(
					self.state, 3), 1)), depth=6), perm = [0, 2, 1]), tf.ones([1,6,7], tf.float32)
			)

			# Reshape the mask into a (42,) so that I don't have to reshape actions/outputs
			self.flat_mask = tf.reshape(self.mask, shape = [1,42])

			# Turn action_taken into one-hot, and then apply the mask to both actor_output and action_taken
			self.action_hot = tf.one_hot(self.action_taken, depth=42)
			self.action_masked = tf.boolean_mask(self.action_hot, self.flat_mask)
			self.P1_actor_masked = tf.boolean_mask(self.P1_actor_output, self.flat_mask)
			self.P2_actor_masked = tf.boolean_mask(self.P2_actor_output, self.flat_mask)

			# Compute the policy loss based on estimated value, and regularization loss
			self.P1_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
					logits = self.P1_actor_masked,
					labels = self.action_masked
			)
			self.P2_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
					logits = self.P2_actor_masked,
					labels = self.action_masked
			)
			self.P1_policy_gradient_loss = tf.reduce_mean(self.P1_cross_entropy_loss)
			self.P2_policy_gradient_loss = tf.reduce_mean(self.P2_cross_entropy_loss)
			self.P1_actor_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_model_P1_variables])
			self.P2_actor_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_model_P2_variables])
			self.P1_actor_total_loss = self.P1_policy_gradient_loss + self.reg_param * self.P1_actor_regularization_loss
			self.P2_actor_total_loss = self.P2_policy_gradient_loss + self.reg_param * self.P2_actor_regularization_loss

			# From the actor loss, calculate the gradients for the actor model
			self.P1_actor_gradients = self.actor_optimizer.compute_gradients(self.P1_actor_total_loss, actor_model_P1_variables)
			self.P2_actor_gradients = self.actor_optimizer.compute_gradients(self.P2_actor_total_loss, actor_model_P2_variables)

			# Compute delta
			self.delta = tf.reduce_sum(self.reward + self.decay*self.next_value - self.estimated_value)

			# Apply delta to the actor gradients:
			for count, (gradient, variable) in enumerate(self.P1_actor_gradients):
				if gradient is not None:
					self.ACTOR_BEFORE = self.P1_actor_gradients[count]
					self.P1_actor_gradients[count] = (gradient * self.delta, variable)
					self.ACTOR_AFTER = self.P1_actor_gradients[count]

			for count, (gradient, variable) in enumerate(self.P2_actor_gradients):
				if gradient is not None:
					self.ACTOR_BEFORE = self.P2_actor_gradients[count]
					self.P2_actor_gradients[count] = (gradient * self.delta, variable)
					self.ACTOR_AFTER = self.P2_actor_gradients[count]

			# Compute the gradients for the critic (difference between discounted reward and estimate squared)
			self.mean_square_loss = tf.reduce_mean(tf.square(self.delta))
			self.critic_regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_model_variables])
			self.critic_total_loss = self.mean_square_loss + self.reg_param * self.critic_regularization_loss

			# Calculate the critic gradients using the total loss
			self.critic_gradients = self.critic_optimizer.compute_gradients(self.critic_total_loss, critic_model_variables)

			# NOTE: Try not multiplying by delta, because that way if delta is negative you have a double negative effect
			#for count, (gradient, variable) in enumerate(self.critic_gradients):
			#	if gradient is not None:
			#		self.critic_gradients[count] = (gradient * self.delta, variable)

			# Clip the gradients to ensure the model doesn't blow up
			for count, (gradient, variable) in enumerate(self.P1_actor_gradients):
				if gradient is not None:
					self.P1_actor_gradients[count] = (tf.clip_by_norm(gradient, self.max_gradient), variable)

			for count, (gradient, variable) in enumerate(self.P2_actor_gradients):
				if gradient is not None:
					self.P2_actor_gradients[count] = (tf.clip_by_norm(gradient, self.max_gradient), variable)

			for count, (gradient, variable) in enumerate(self.critic_gradients):
				if gradient is not None:
					self.critic_gradients[count] = (tf.clip_by_norm(gradient, self.max_gradient), variable)
			
		# Apply the calculated gradients
		with tf.name_scope("train_actor_critic"):
			# Use the gradients calculated earlier to update the actor network
			self.train_op_P1 = self.actor_optimizer.apply_gradients(self.P1_actor_gradients)
			self.train_op_P2 = self.actor_optimizer.apply_gradients(self.P2_actor_gradients)
			self.train_critic = self.critic_optimizer.apply_gradients(self.critic_gradients)

	
	def sampleAction(self, state, player):

		def softmax(y):
			maxy = np.amax(y)
			e = np.exp(y - maxy)
			return e / np.sum(e)

		# Determine action scores by running the actor model
		if player == 1:
			action_scores = self.session.run(self.P1_pre_action_scores, {self.state: state})[0]
		else:
			action_scores = self.session.run(self.P2_pre_action_scores, {self.state: state})[0]

		# Determine the valid moves, and set the log-likelihood of invalid moves to 0
		valid_moves = np.reshape(np.transpose(np.equal.outer(5 - np.count_nonzero(np.sum(
				state, 3), 1), np.arange(6)).astype(np.int), [0, 2, 1]), [42,]
		)
		action_scores[valid_moves==0] = float("-inf")

		# Perform a softmax operation on the valid action scores
		valid_scores = softmax(action_scores)

		# Randomly choose an action using the probabilities over valid moves
		action = np.random.choice(range(42), p=valid_scores)

		return action

	
	def updateP1Model(self, state, action, reward, next_state, show_game):
	
		### NOTE: NOTE: NOTE: THE GRADIENTS ARE WORKING PROPERLY, AND THE
		### MASKING IS WORKING PROPERLY. THE PROBLEM IS OUTPUTTING THE LIST
		### OF SEVEN POSSIBLE MOVES FOR THE HUMAN OPERATOR. THESE NUMBERS
		### ARE IN ORDER OF ROWS, /NOT/ IN ORDER OF COLUMNS. 
		### [0 1 0 0 0 0 0] MAY /NOT/ REFER TO THE SECOND COLUMN, JUST THE
		### SECOND AVAILABLE POSITION WHEN SCANNING FROM TOP LEFT TO BOTTOM
		### RIGHT. BUT THE MODEL SHOULD BE WORKING PROPERLY INTERNALLY

		estv1 = 0
		estv2 = 0
		next1 = 0
		delta = 0

		checker = 0
		if reward == -1:
			reward = 0
			checker = 1

		action = np.array([action], dtype=np.int32)
		reward = np.array([reward], dtype=np.float32)

		# Perform the training update
		_, _, estv1, next1, delta, all_actions, the_action = self.session.run([
				self.train_op_P1,
				self.train_critic,
				self.estimated_value,
				self.next_value,
				self.delta,
				self.P1_actor_masked,
				self.action_masked
		], {
				self.state: state,
				self.next_state: next_state,
				self.action_taken: action,
				self.reward: reward
		})	

		if show_game == 1:
			# Recheck the values
			estv2 = self.session.run([
					self.estimated_value
			], {
					self.state: state,
					self.next_state: next_state,
					self.action_taken: action,
					self.reward: reward
			})[0]

			# TURN FEED BOARD INTO REGULAR BOARD FOR DISPLAY PURPOSES
			# This is for player 1, because this is "update P1 model"
			Boardstate = np.zeros((6,7), dtype=int)
			for row in range(state.shape[1]):
				for col in range(state.shape[2]):
					if state[0, row, col, 0] == 1:
						Boardstate[row, col] = 1
					if state[0, row, col, 1] == 1:
						Boardstate[row, col] = 2

			# Print board, and where it goes, 
			# then what it thinks to do before and after updating
			print("Game num: {}".format(self.game_num))
			print("Board:")
			print(Boardstate)
			#print("Actor output:")
			#print(all_actions)
			#print(the_action)
			print("Action chosen: {}".format((action%7)+1))
			print("Critic evaluation of Board: {}".format(estv1))
			print("Next critic evaluation: {}".format(next1))
			print("Delta value: {}".format(delta))
			print("New critic evaluation: {}".format(estv2))
			
		if reward != 0:
			self.game_num += 1
		if checker == 1:
			self.game_num += 1

	def updateP2Model(self, state, action, reward, next_state, show_game):
	
		### NOTE: NOTE: NOTE: THE GRADIENTS ARE WORKING PROPERLY, AND THE
		### MASKING IS WORKING PROPERLY. THE PROBLEM IS OUTPUTTING THE LIST
		### OF SEVEN POSSIBLE MOVES FOR THE HUMAN OPERATOR. THESE NUMBERS
		### ARE IN ORDER OF ROWS, /NOT/ IN ORDER OF COLUMNS. 
		### [0 1 0 0 0 0 0] MAY /NOT/ REFER TO THE SECOND COLUMN, JUST THE
		### SECOND AVAILABLE POSITION WHEN SCANNING FROM TOP LEFT TO BOTTOM
		### RIGHT. BUT THE MODEL SHOULD BE WORKING PROPERLY INTERNALLY

		estv1 = 0
		estv2 = 0
		next1 = 0
		delta = 0

		checker = 0
		if reward == -1:
			reward = 0
			checker = 1

		action = np.array([action], dtype=np.int32)
		reward = np.array([reward], dtype=np.float32)
		
		# Perform the training update
		_, _, estv1, next1, delta, all_actions, the_action = self.session.run([
				self.train_op_P2,
				self.train_critic,
				self.estimated_value,
				self.next_value,
				self.delta,
				self.P2_actor_masked,
				self.action_masked
		], {
				self.state: state,
				self.next_state: next_state,
				self.action_taken: action,
				self.reward: reward
		})	

		if show_game == 1:
			# Recheck the values
			estv2 = self.session.run([
					self.estimated_value
			], {
					self.state: state,
					self.next_state: next_state,
					self.action_taken: action,
					self.reward: reward
			})[0]

			# TURN FEED BOARD INTO REGULAR BOARD FOR DISPLAY PURPOSES
			# This is for player 2, because this is "update P2 model"
			Boardstate = np.zeros((6,7), dtype=int)
			for row in range(state.shape[1]):
				for col in range(state.shape[2]):
					if state[0, row, col, 0] == 1:
						Boardstate[row, col] = 2
					if state[0, row, col, 1] == 1:
						Boardstate[row, col] = 1

			# Print board, and where it goes, 
			# then what it thinks to do before and after updating
			print("Game num: {}".format(self.game_num))
			print("Board:")
			print(Boardstate)
			#print("Actor output:")
			#print(all_actions)
			#print(the_action)
			print("Action chosen: {}".format((action%7)+1))
			print("Critic evaluation of Board: {}".format(estv1))
			print("Next critic evaluation: {}".format(next1))
			print("Delta value: {}".format(delta))
			print("New critic evaluation: {}".format(estv2))

		if reward != 0:
			self.game_num += 1
		if checker == 1:
			self.game_num += 1
