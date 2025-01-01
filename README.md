# Snake-Game-Ai
.Reinforcement learning
its an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cummulative reward.

RL is teaching a software agent how to behave in environment by telling it how good it's doing. 

.Deep q learning
approach extends reinforcement learning by using a deep neural network to predict the actions.


.Agent:
-game(pygame)
play_step(action)
	->reward, gane_over,score

-Model(pyTorch)
linear_Qnet(DQN)
-model.predict(state)
	->action



.Training:
state=get_state(game)
action=get_move(state):
	model.predict()
reward ,game_over,score=game.play_step(action)
new_state=get_state(game)
remember
model.train()



.rewards:
eat food: +10
game over:-10
else:0

.Action
[1,0,0]->straight
[0,1,0]->right turn
[0,0,1]->left turn


.state:
[danger straight, danger right,dangerleft ,direction left, direction right, direction up , direction down, food left ,foodright,food up, food down]

ex:
[anger straight, danger right,dangerleft ,
direction left, direction right, direction up , direction down, food left ,foodright,food up, food down]
 so:
[0,0,0,
0,1,0,0,
0,1,0,1]



Deep Q learning:
Q value: Quality of action

init q value(=init model)
choose action(model.predit(state))<--------(repeat)
perform action                            |
measure reward                            |
update Qvalue(+train model)---------------|

BELLMAN EQUATION:
newQ(s,a)=Q(s,a)+ alpha[R(s,a)+ gama maxQ'(s',a')-Q(s,a)]
The Bellman equation expresses a relationship between the value of a decision problem at a certain point in time and the value of its subproblems at subsequent points in time.

In reinforcement learning, the Bellman equation is typically used to describe the relationship between the value function of a state and the value functions of its successor states.

simplified:
Q= model.predict(state0)
Qnew= R(Reward)+ gama.max(Q(state1))

loss function:
loss=(Qnew-Q)^2
       |
       ----->mean squared error

pip3 install torch torchvision
pip install matplotlib ipython
