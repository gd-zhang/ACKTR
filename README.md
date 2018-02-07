# ACKTR
An implementation of `ACKTR` in TensorFlow. ACKTR is one of the current SOTA policy gradient methods. Openai provides included the code for ACKTR in [baselines](https://github.com/openai/baselines). However, these baselines are difficult to understand and modify, especially kfac.py. So, I made the ACKTR based on [A2C](https://github.com/MG2033/A2C) and [Tensorflow KFAC](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/kfac).

### Pros
1. Using tf.contrib.kfac which is easier to follow and well-written.
2. Support for Tensorboard visualization per running agent in an environment.
3. Support for different policy networks in an easier way.
4. Support for environments other than OpenAI gym in an easy way.

### Cons
1. Only support sync computation which makes it slower than openai baseline. (I will update the code regularly and hopefully support async compuation soon.)
2. You need to tune the hyperparamters, especially hyperparamters for KFAC optimizer. (I'm tunning very hard and will update the best configuration regularly.)

## Actor Critic using Kronecker-Factored Trust Region (ACKTR)
Actor Critic using Kronecker-Factored Trust Region was introduced in [ACKTR](https://arxiv.org/pdf/1708.05144.pdf). It used a recently proposed technique called [K-FAC](https://arxiv.org/abs/1503.05671) (a very strong optimzer) for actor-critic methods. It shows 2- to 3-fold improvement in sample efficiency.

### Environments Supported
This implementation allows for using different environments. It's not restricted to OpenAI gym environments. If you want to attach the project to another environment rather than that provided by gym, all you have to do is to inherit from the base class `BaseEnv` in `envs/base_env.py`, and implement all the methods in a plug and play fashion.

The methods that should be implemented in a new environment class are: 
1. `make()` for creating the environment and returning a reference to it.
2. `step()` for taking a step in the environment and returning a tuple (observation images, reward float value, done boolean, any other info).
3. `reset()` for resetting the environment to the initial state.
4. `get_observation_space()` for returning an object with attribute tuple `shape` representing the shape of the observation space.
5. `get_action_space()` for returning an object with attribute `n` representing the number of possible actions in the environment.
6. `render()` for rendering the environment if appropriate.

### Policy Networks Supported
This implementation comes with the basic CNN policy network from OpenAI baseline (using 32 filters in last conv layer). However, it supports using different policy networks. All you have to do is to inherit from the base class `BasePolicy` in `models\base_policy.py`, and implement all the methods in a plug and play fashion again :D (See the CNNPolicy example class). You also have to add the name of the new policy network class in `models\model.py\policy_name_parser()` method.

### Tensorboard Visualization
This implementation allows for the beautiful Tensorboard visualization. It displays the time plots per running agent of the two most important signals in reinforcement learning: episode length and total reward in the episode. All you have to do is to launch Tensorboard from your experiment directory located in `experiments/`.
```
tensorboard --logdir=experiments/breakout_config/summaries
```

### Video Generation
During training, you can generate videos of the trained agent acting (playing) in the environment. This is achieved by changing `record_video_every` in the configuration file from -1 to the number of episodes between two generated videos. Videos are generated in your experiment directory.

During testing, videos are generated automatically if the optional `monitor` method is implemented in the environment. As for the gym included environment, it's already been implemented.

### Run
```
python main.py --config config/breakout_config.json
```

## Reference Repository
[OpenAI Baselines](https://github.com/openai/baselines)

[A2C](https://github.com/MG2033/A2C)
