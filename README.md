# Connect 4 Self-Play

This code is an implementation of the self-play general reinforcement learning algorithm described in [this paper](https://arxiv.org/pdf/1712.01815.pdf) by David Silver *et al*. This algorithm uses a single neural network that accepts the boardstate **s** as an input, and outputs a vector of move probabilities **p** with components **p<sub>a</sub>** = Pr(**a**|**s**), for each action **a**, and also a scalar value **v** that is an estimate of the expected outcome **z** of the game, given the position.

This was written for fun, to explore the capabilities of machine learning. Connect 4 is a solved game; I was interested in watching a program teach itself to play, *not* in finding the "perfect game".

## Running the Code

To run the code, make sure the following files are all in the same folder:
```
run_connect4bot_MCTS.py
connect4bot_MCTS.py
connect4game.py
```
Then, run the code "run_connect4bot_MCTS.py". This code contains the main body of the program, including setting parameters, the Monte Carlo tree search, saving the model, and running test games to monitor the progress of the neural network.

## Connect 4 Rules

![alt text](http://www.boardgamecapital.com/game_images/connect-four.jpg "Connect Four")

In Connect 4, two players take turns dropping checkers of their color into one of seven columns (pictured). The first player to get four of their colored checkers in a row, either horizontally, vertically, or diagonally, wins the game. 

## General Description of the Code

The current version of this code works by combining Monte Carlo tree search (MCTS) with a neural network. Given any boardstate, a starting move is decided upon by maximizing the following function over all legal moves (which is a combination of a function in the Silver *et al*. paper above, and an *upper confidence bound* (UCB) action selection technique):
```
a*Q + b*P/(1+N) + c*D/(1+N) + d*sqrt(t)/N
```
where:
* [*a*, *b*, *c*, *d*] are constants; 
* *Q* is a running average of game outcomes from the current player's perspective for each legal move; 
* *P* is the output of the neural network corresponding to each legal move, representing a "preference" for each move; 
* *D* is Dirichlet noise added to each legal move, to aid in exploration; 
* *N* is the visit count for each legal move; and 
* *t* is the total number of Monte Carlo rollouts performed over all legal moves.

As each rollout (Monte Carlo search from the current boardstate to the end of the game) is completed, *Q* and *N* are updated for the move that was chosen, and *t* is incremented. At the start of the tree search, *N* is 0 for each legal move, and so the "d" term is *Inf*. This causes each move to be selected at least once. At this point, future choices for the tree search are directed largely by how well that legal move performed (*Q*) and the neural network's preference (*P*). As the number of rollouts increases, *Q* becomes a better estimate of the actual game outcome following each legal move, the influence of the neural network's preference (*P*) decreases, and moves that have not been frequently visited (high *t*, low *N*) are more likely to be selected. As a large number of rollouts are completed the visit count becomes mostly representative of how well the player performed following each legal move, with higher visit counts meaning that the move likely had good performance in past rollouts.

It should also be noted that moves in each rollout that are *not* the starting move are decided by maximizing a simpler version of the above function:
```
a*Q + b*P/(1+N)
```
In this function, as compared to the more involved function above, the influence of the neural network's preference (*P*) is higher, but still decreases in favor of *Q* as more rollouts are performed and the estimate becomes a better estimate of the actual game outcome.

Values for *Q*, *P*, and *N* are stored for each legal move, for each boardstate that is visited during the tree search.

Following the MCTS, the legal move with the highest visit count *N* is selected as the final choice. This move is performed, and a new MCTS is started from the opposite player's perspective, starting from the new boardstate. This process of running a large number of Monte Carlo rollouts and then selecting a final move is counted as a single datapoint for training the neural network: the starting boardstate is the input, and the distribution of *N* over all legal moves is the "target" for what the neural network should output. In this way, the neural network is being trained to output a distribution over legal moves that favors moves with the highest win-rate.

## More Specific Code Notes

This section covers more specific notes about how certain portions of code work.

### Neural Network Input/Output

In order to turn the Connect 4 board into an input for the neural network, the board (normally size [6, 7]) is re-formatted into a one-hot tensor of size [6, 7, 2]. The first two dimensions of the tensor correspond to the row and column, respectively, on the board, and the final dimension is either [1, 0] if the current player has a checker in that position, [0, 1] if the opposing player has a checker in that position, or [0, 0] if the position is empty.

The output of the neural network must have constant dimensions, but the legal moves are dependent on the boardstate. To get around this problem, the output of the network is simply a vector of size [6x7], corresponding to each possible board position. For a given boardstate, outputs that correspond to illegal moves are masked (set to 0), and a softmax function is performed over the remaining legal moves. There is an additional scalar output which corresponds to the network's prediction of winning or losing given the board.

### Storing Boardstates and Counting Rollouts

In order to store information for each boardstate, boardstates are first transformed into a unique string to identify the board:
```
def concatenate_state_data(Board):
	result = ''
	for row in Board:
		for element in row:
			result += str(element)
	return result
```
This string is then indexed by a number (starting with 0, incrementing by 1 for each new unique boardstate) so that information can be easily connected with boardstates. The stored information is:
* **P**, which is the neural network output given the specific boardstate as an input. This is stored so that it only needs to be calculated the first time a unique board is seen.
* **Q**, which is the running average winrate after moving in each legal move. This is stored so that it may be updated each time a boardstate is passed through.
* **N**, which is the visit count of each legal move for the specific boardstate. This is stored for calculating moves during Monte Carlo rollouts.

All of this information is maintained for each unique boardstate that is visited during Monte Carlo rollouts. This serves several purposes. First, redundant calls to the neural network are removed, speeding up the code. Second, a rollout starting from the first move in a game will almost certainly pass through boardstates that will later be encountered by rollouts from future moves. These future rollouts can then make use of information gathered during previous rollouts (i.e., **Q**), and can make better-informed choices on where to move (i.e., as **Q** gets more accurate, **N** also increases, which means **Q** becomes more influential as compared to **P**/(1+**N**)).

The number of visited unique states almost always increases as more rollouts are completed, but as moves are performed ("real" moves, which are selected after each batch of MC rollouts), previously-visited boardstates (which cannot re-occur in the same game) and now-impossible boardstates (pieces cannot be removed, so conflicts may appear) and their associated **P**, **Q**, and **N** vectors are removed. This cleaning is performed using the unique strings associated with each board, as follows:
```
def index_cleaner(Board, Board_index, pred_index, P_index, Q_index, N_index, V_index):
	Board_string = concatenate_state_data(Board)

	unwanted_list = []

	for index, index_string in enumerate(Board_index):
		if len([c for c,d in zip(Board_string, index_string) if c!="0" and c!=d]) > 0:
			unwanted_list.append(index)

	Board_index = [i for j,i in enumerate(Board_index) if j not in unwanted_list]
	pred_index = [i for j,i in enumerate(pred_index) if j not in unwanted_list]
	P_index = [i for j,i in enumerate(P_index) if j not in unwanted_list]
	Q_index = [i for j,i in enumerate(Q_index) if j not in unwanted_list]
	N_index = [i for j,i in enumerate(N_index) if j not in unwanted_list]
	V_index = [i for j,i in enumerate(V_index) if j not in unwanted_list]

	return Board_index, pred_index, P_index, Q_index, N_index, V_index
```
This process frees up memory and speeds up the code. 

### Testing Neural Network Progress

In order to check whether the neural network is improving, after each update, the neural network plays 1000 games of Connect 4 against a completely random opponent. 500 games are played with the network as Player 1, and 500 games as Player 2. The moves for the neural network are decided purely based on the network output: no Monte Carlo searches allowed. Moves are also picked greedily: given any boardstate, whatever legal move has the highest value is the move that is selected. For the "completely random opponent", moves are simply selected by picking a random (legal) column to place a checker in.

Of course, a random player is not a very formidable opponent for any human player. For the neural network, moving in the same column over and over is a decent enough strategy, since the random opponent has no concept of "blocking" potential wins, but it is not a strong enough strategy to *always* win. Comparatively, almost any human could have a near-perfect winrate against a random opponent (with the exceptions being the rare chance of the random player playing perfectly, and the human getting overconfident and missing a "block" for enough turns that the random player finds the win). Given enough training time, the neural network should eventually achieve a perfect or near-perfect winrate. 
