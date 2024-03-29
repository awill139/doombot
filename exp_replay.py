import numpy as np
from collections import namedtuple, deque

Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

class NStepProg:
    def __init__(self, env, brain, n_steps):
        self.brain = brain
        self.rewards = []
        self.env = env
        self.n_steps = n_steps

    def __iter__(self):
        state = self.env.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.brain(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_steps + 1:
                history.popleft()
            if len(history) == self.n_steps + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_steps + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                history.clear()

        def reward_steps(self):
            reward_steps = self.rewards
            self.rewards = []
            return reward_steps

class ReplayMem:
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()
    
    def sample_batches(self, batch_size):
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while(ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
    
    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)
            self.buffer.append(entry)
            samples -= 1
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()