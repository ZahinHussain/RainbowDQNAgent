import numpy as np
import random
from collections import deque

'''

SumTree is a data strucuture which the childs adds to the sum of the perent

'''
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity  # Maximum number of elements to store
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # Binary tree structure
        self.data_pointer = 0

    def update(self, tree_idx, priority):
        # Update a specific node and propagate the change
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    v -= self.tree[left_child]
                    parent_idx = right_child
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self):
        return self.tree[0]
        
class NStepPrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape,
                 n_step=3, gamma=0.99, alpha=0.6, beta=0.4):

        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Replay arrays
        self.obs       = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs  = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions   = np.zeros((capacity,), dtype=np.int64)
        self.rewards   = np.zeros((capacity,), dtype=np.float32)
        self.dones     = np.zeros((capacity,), dtype=np.float32)
        self.discounts = np.zeros((capacity,), dtype=np.float32)

        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.tree = SumTree(capacity)

        # N-step
        self.n_step = n_step
        self.gamma  = gamma
        self.nstep_buffer = deque(maxlen=n_step)

    def get_nstep(self):
        R, next_s, done = self.nstep_buffer[-1][2:]

        discount = 1.0

        for (_, _, r, ns, d) in reversed(list(self.nstep_buffer)[:-1]):
            R = r + self.gamma * R * (1 - d)
            discount *= self.gamma
            if d:
                next_s = ns
                done = True
                break

        return R, next_s, done, discount

    def _store(self, s, a, Rn, ns, done, g_n):
        # Index in the replay buffer where the transition will be written
        # (circular buffer: overwrites old data when full)
        idx = self.ptr

        # Store transition
        self.obs[idx]       = s      
        self.actions[idx]   = a      
        self.rewards[idx]   = Rn     
        self.next_obs[idx]  = ns     
        self.dones[idx]     = done   
        self.discounts[idx] = g_n    

        # Convert replay-buffer index to leaf index in the SumTree
        tree_idx = idx + self.capacity - 1

        # Assign initial priority to new transition
        # Use current max priority so new samples are guaranteed to be seen
        # Apply alpha here to match PER definition: p_i^alpha
        priority = self.max_priority ** self.alpha

        # Update the SumTree leaf and propagate change up the tree
        self.tree.update(tree_idx, priority)

        # Move write pointer forward (wrap around when reaching capacity)
        self.ptr = (self.ptr + 1) % self.capacity

        # Track current buffer size (never exceeds capacity)
        self.size = min(self.size + 1, self.capacity)

    def store(self, s, a, r, ns, done):
        # Add the current transition to the temporary n-step buffer
        self.nstep_buffer.append((s, a, r, ns, done))

        # Case 1: n-step buffer is full
        if len(self.nstep_buffer) == self.n_step:

            Rn, next_s, dn, g_n = self.get_nstep()

            # Use the *first* (state, action) in the n-step window
            # This is the transition that will be stored in replay
            s0, a0 = self.nstep_buffer[0][:2]

            # Store the n-step transition into replay buffer + SumTree
            self._store(s0, a0, Rn, next_s, dn, g_n)

        # Case 2: episode ended
        # When an episode ends, there may still be remaining transitions
        # in the n-step buffer that haven't been stored yet.
        if done:
            while len(self.nstep_buffer) > 0:

                # Compute the n-step return for the remaining transitions
                # (these will be shorter than n steps)
                Rn, next_s, dn, g_n = self.get_nstep()

                # Store the earliest transition still in the buffer
                s0, a0 = self.nstep_buffer[0][:2]
                self._store(s0, a0, Rn, next_s, dn, g_n)

                # Remove the stored transition and continue
                self.nstep_buffer.popleft()

    def sample(self, batch_size):
        # Total sum of all priorities (root of the SumTree)
        total = self.tree.total_priority()

        # Divide the total priority range into equal segments for sampling
        segment = total / batch_size

        idxs = []        # indices into replay buffer
        priorities = []  # sampled priorities 

        for i in range(batch_size):
            # Sample a value uniformly from the i-th segment
            mass = random.uniform(segment * i, segment * (i + 1))

            # Traverse the SumTree to find the leaf corresponding to 'mass'
            _, priority, data_idx = self.tree.get_leaf(mass)

            idxs.append(data_idx)
            priorities.append(priority)

        # Convert to numpy arrays
        idxs = np.array(idxs, dtype=np.int32)
        priorities = np.array(priorities, dtype=np.float32)

        # Probability of sampling each transition
        probs = priorities / total

        # Importance-sampling weights & normalise 
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        return (
            self.obs[idxs],        
            self.actions[idxs],    
            self.rewards[idxs],    
            self.next_obs[idxs],  
            self.dones[idxs],      
            self.discounts[idxs], 
            idxs,                  
            weights,               
        )

    def update_priorities(self, idxs, td_errors):
        # Convert TD-errors into priorities
        # Priorities must be non-negative, 1e-6: avoids zero priority, self.alpha controls how strongly prioritization is used
        prios = (np.abs(td_errors) + 1e-6) ** self.alpha

        # Update each sampled transition's priority in the SumTree
        for idx, p in zip(idxs, prios):
            tree_idx = idx + self.capacity - 1

            # Update the leaf node and propagate the change up the tree
            self.tree.update(tree_idx, p)

            # Update max priority
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.size

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self):
        return self.size
