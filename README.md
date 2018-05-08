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
where [*a*, *b*, *c*, *d*] are constants; *Q* is a running average of game outcomes from the current player's perspective for each legal move; *P* is the output of the neural network corresponding to each legal move, representing a "preference" for each move; *D* is Dirichlet noise added to each legal move, to aid in exploration; *N* is the visit count for each legal move; and *t* is the total number of Monte Carlo rollouts performed over all legal moves.

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

The output of the neural network must have constant dimensions, but the legal moves are dependent on the boardstate. To get around this problem, the output of the network is simply a vector of size [6*7], corresponding to each possible board position. For a given boardstate, outputs that correspond to illegal moves are masked (set to 0), and a softmax function is performed over the remaining legal moves. There is an additional scalar output which corresponds to the network's prediction of winning or losing given the board.
