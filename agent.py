import tensorflow as tf
from memory import Memory
import numpy as np
import base64, io, time, gym
import IPython, functools
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import mitdeeplearning as mdl

def aggregate_memories(memories):
    batch_memory = Memory()
    for one_memory in memories:
        for step in zip(one_memory.observations, one_memory.actions, one_memory.rewards):
            batch_memory.add_to_memory(*step)
    return batch_memory

memory = Memory()

def create_cartpole_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=2, activation=None)
    ])
    return model


def choose_action(model, observation, single=True):
    observation = np.expand_dims(observation, axis=0) if single else observation
    logits = model.predict(observation)
    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()
    return action[0] if single else action


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

env = gym.make("CartPole-v0")
env.seed(1)
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
cartpole_model = create_cartpole_model()
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()

for i_episode in range(10):

    plotter.plot(smoothed_reward.get())

    # Restart the environment
    observation = env.reset()
    memory.clear()

    while True:
        # using our observation, choose an action and take it in the environment
        action = choose_action(cartpole_model, observation)
        next_observation, reward, done, info = env.step(action)
        # add to memory
        memory.add_to_memory(observation, action, reward)

        # is the episode over? did you crash or do so well that you're done?
        if done:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)
            smoothed_reward.append(total_reward)

            # initiate training - remember we don't know anything about how the
            #   agent is doing until it has crashed!
            train_step(cartpole_model, optimizer,
                       observations=np.vstack(memory.observations),
                       actions=np.array(memory.actions),
                       discounted_rewards=discount_rewards(memory.rewards))

            # reset the memory
            memory.clear()
            break
        # update our observatons
        observation = next_observation