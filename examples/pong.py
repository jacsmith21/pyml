import os

import gym
import tensorflow as tf
import tensortools as tt
import numpy as np


def preprocess(image):
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()


# noinspection PyShadowingNames
def discount_rewards(rewards, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for time in reversed(range(len(rewards))):
        if rewards[time] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)

        running_add = running_add * gamma + rewards[time]
        discounted_r[time] = running_add

    return discounted_r


# noinspection PyShadowingNames
def build_network(n_pixels, n_units, learning_rate=1e-3, decay=0.99):
    pixels = tf.placeholder(tf.float32, [None, n_pixels])
    actions = tf.placeholder(tf.float32, [None, 1])
    rewards = tf.placeholder(tf.float32, [None, 1])

    hidden = tt.ops.fully_connected(pixels, n_units, use_bias=False, activation=tf.nn.relu,
                                    initializer=tf.contrib.layers.xavier_initializer())
    logits = tt.ops.fully_connected(hidden, 1, use_bias=False, initializer=tf.contrib.layers.xavier_initializer())

    probability = tf.nn.sigmoid(logits)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=actions, logits=logits)
    loss = tf.reduce_sum(rewards * cross_entropy)

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay).minimize(loss, global_step=global_step)

    tf.summary.histogram("hidden", hidden)
    tf.summary.histogram("logits", logits)
    tf.summary.histogram("probability", probability)

    return pixels, actions, rewards, probability, optimizer


n_pixels = 80*80
n_hidden_units = 200
batch = 10
checkpoint_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'out')

pixels_ph, actions_ph, rewards_ph, probabilities_op, optimizer = build_network(n_pixels, n_hidden_units)

env = gym.make("Pong-v0")
state = env.reset()
saver = tf.train.Saver()
hooks = [tt.hooks.GlobalStepIncrementor()]
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks) as sess:
    frames, actions, rewards, discounted_rewards = [], [], [], []
    episode, step = 0, 0
    running_reward = 0
    prev_frame = None
    while not sess.should_stop():
        curr_frame = preprocess(state)
        frame = curr_frame - prev_frame if prev_frame is not None else np.zeros_like(curr_frame)
        prev_frame = curr_frame

        probabilities = sess.run(probabilities_op, feed_dict={pixels_ph: np.reshape(frame, [1, n_pixels])})
        probability = probabilities[0]  # there's only one in the batch
        probability = probability[0]  # there's only one actual probability
        action = 1 if np.random.uniform() < probability else 0

        state, reward, done, info = env.step(action)
        observation = env.reset()  # getting ready for the next iteration

        frames.append(frame)
        actions.append(action)
        rewards.append(reward)

        if not done:
            continue

        rewards = discount_rewards(rewards)
        rewards = tt.utils.normalize(rewards)
        discounted_rewards.extend(rewards.tolist())
        rewards = []

        running_reward = 0.99 * running_reward + 0.01 * (sum(rewards))
        tf.summary.scalar('running_reward', running_reward)

        episode += 1

        if episode % batch != 0:
            continue

        # Ok, it's time to actually train!
        # Lets go!

        # batch everything!
        frames = np.vstack(frames)
        actions = np.vstack(actions)
        discounted_rewards = np.vstack(discounted_rewards)

        sess.run(optimizer, feed_dict={pixels_ph: frames, actions_ph: actions, rewards_ph: discounted_rewards})

        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=step)
        print("episode: {}, step: {}, reward: {}".format(episode, step, running_reward))

        frames, actions, discounted_rewards = [], [], []

        step += 1
