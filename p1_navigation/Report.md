## Navigation Project Report

Technique Used:
- Deep Q-Network
- Double DQN
- Duelling Network
- Priority Replay Buffer (maybe not implemented correctly, it doesn't improve score)

During the project, the Duelling Network gives the best performance boost, while
the Double DQN gives little or no improvement.
Another observation is the performance doesn't increase when scaling up the network size more.

### Results
The training will terminate once average score reaches 12.

The network converges in 370 episodes

The score per episode during training:

![](dqn_score.png)


### Network Architecture using Duelling Network
The network architecture is simple, with one hidden fully connected layer of 128 neurons, and
two branches after it. One fully connected layer to calculate the 1 state value, and another 
fully connected layer to calculate the 4 action value offsets.

The result of the two heads are summed up by the following equation to form the final action values.
```python
(action_value_offsets - mean(action_value_offsets)) + state_value
```

 ![](pics/dqn_net.png)


### Demo
Runing the trained model with exploration epsilon = 0.01
![](pics/dqn_demo.gif)


### Future work
- We can see in the 3rd episode from the demo, the agent stuck in one situation
and is not able to jump out. It is worth investigating why and how to solve this problem.

- The priority reply buffer is not implemented properly, will be fixed in the future.

