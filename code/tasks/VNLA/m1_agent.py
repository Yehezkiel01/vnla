# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division

import json
import os
import sys
import numpy as np
import random
import time
import math
import collections

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from agent import BaseAgent
from oracle import make_oracle
from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent

# DQN HYPERPARAMETER

## Reward Shaping
SUCCESS_REWARD = 25
FAIL_REWARD = 0
STEP_REWARD = -1
ASK_REWARD = -2

## DQN Buffer
BUFFER_LIMIT = 1000
MIN_BUFFER_SIZE = 100

# Data structure to accumulate and preprocess training data before being inserted into our DQN Buffer for experience replay
class Transition:
    ASKING_ACTIONS = [1, 2, 3, 4]       # 0, 5, 6 are considered non-asking actions

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def _clone_tensor(self, t):
        return None if t is None else t.clone().detach()

    def _clone_states(self, states):
        a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov = states

        a_t_copy = self._clone_tensor(a_t)
        q_t_copy = self._clone_tensor(q_t)
        f_t_copy = self._clone_tensor(f_t)

        # Decoder_h is a tuple of tensor, thus it needs special handling
        decoder_h_copy = None
        if decoder_h is not None:
            decoder_h_copy = (decoder_h[0].clone().detach(), decoder_h[1].clone().detach())

        ctx_copy = self._clone_tensor(ctx)
        seq_mask_copy = self._clone_tensor(seq_mask)
        nav_logit_mask_copy = self._clone_tensor(nav_logit_mask)
        ask_logit_mask_copy = self._clone_tensor(ask_logit_mask)
        b_t_copy = self._clone_tensor(b_t)
        cov_copy = self._clone_tensor(cov)

        return (a_t_copy, q_t_copy, f_t_copy, decoder_h_copy, ctx_copy, seq_mask_copy,
                nav_logit_mask_copy, ask_logit_mask_copy, b_t_copy, cov_copy)

    def add_states(self, states):
        self.states = self._clone_states(states)

    def add_next_states(self, next_states):
        self.states = self._clone_states(next_states)

    def add_filter(self, filter):
        self.filter = np.copy(filter)

    def add_is_done(self, is_done):
        self.is_done = np.copy(is_done)

    def add_is_success(self, is_success):
        self.is_success = np.copy(is_success)

    def add_actions(self, actions):
        self.actions = actions.clone().detach()

    def compute_reward_shaping(self):
        self.rewards = np.empty(self.batch_size)

        for i in range(self.batch_size):
            if (self.filter[i]):
                continue

            if (self.is_done[i]):
                self.rewards[i] = SUCCESS_REWARD if self.is_success[i] else FAIL_REWARD
            else:
                if self.actions[i] in Transition.ASKING_ACTIONS:
                    self.rewards[i] = ASK_REWARD
                else:
                    self.rewards[i] = STEP_REWARD

    def to_list(self):
        # Since all the states, rewards, actions, etc are grouped in batch, we need to divide them before inserting them to the queue
        # In addition we need to filter some of them which are invalid
        experiences = []
        for i in range(self.batch_size):
            if (self.filter[i]):
                continue

            state = tuple([None] * len(self.states))
            for j in range(len(self.states)):
                state[j] = self.states[j][i]

            action = self.actions[i]
            reward = self.rewards[i]

            next_state = tuple([None] * len(self.next_states))
            for j in range(len(self.next_states)):
                next_state[j] = self.next_states[j][i]

            is_done = self.is_done[i]

            experiences.append((state, action, reward, next_state, is_done))
        return experiences

# Maintain past experiences for DQN experience replay
class ReplayBuffer():
    def __init__(self, buffer_limit=BUFFER_LIMIT):
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque()

    def push(self, experience):
        '''
        Input:
            * `experience`: A tuple of (state, action, reward, next_state, is_done).
                            if the buffer is full, the oldest experience will be discarded.
        Output:
            * None
        '''
        if self.__len__() == self.buffer_limit:
            self.buffer.popleft()           # Remove the oldest experience

        self.buffer.append(experience)

    def push_multiple(self, experiences):
        '''
        Input:
            * `experiences`: An array of tuple of (state, action, reward, next_state, is_done).
                            if the buffer is full, the oldest experience will be discarded.
        Output:
            * None
        '''

        for experience in experiences:
            self.push(experience)

    def sample(self, batch_size):
        '''
        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        transitions = random.sample(self.buffer, batch_size)

        states = torch.from_numpy(np.array([t.state for t in transitions]))
        actions = torch.from_numpy(np.array([t.action for t in transitions]))
        rewards = torch.from_numpy(np.array([t.reward for t in transitions]))
        next_states = torch.from_numpy(np.array([t.next_state for t in transitions]))
        dones = torch.from_numpy(np.array([t.done for t in transitions]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)

# This agent is a DQN Trainer
class M1Agent(VerbalAskAgent):

    def __init__(self, model, hparams, device, train_evaluator):
        super(M1Agent, self).__init__(model, hparams, device)

        # This evaluator will only be used if self.is_eval is False.
        # The evaluator is necessary for the RL to award the correct reward to the agent
        self.train_evaluator = train_evaluator
        self.buffer = ReplayBuffer(BUFFER_LIMIT)
        # TODO: Freeze model except for ask_predictor

    def compute_states(self, batch_size, obs, queries_unused, existing_states):
        # Unpack existing states
        a_t, q_t, decoder_h, ctx, seq_mask, cov = existing_states

        # Mask out invalid actions
        nav_logit_mask = torch.zeros(batch_size,
                                        AskAgent.n_output_nav_actions(), dtype=torch.uint8, device=self.device)
        ask_logit_mask = torch.zeros(batch_size,
                                        AskAgent.n_output_ask_actions(), dtype=torch.uint8, device=self.device)

        nav_mask_indices = []
        ask_mask_indices = []
        for i, ob in enumerate(obs):
            if len(ob['navigableLocations']) <= 1:
                nav_mask_indices.append((i, self.nav_actions.index('forward')))

            if queries_unused[i] <= 0:
                for question in self.question_pool:
                    ask_mask_indices.append((i, self.ask_actions.index(question)))

        nav_logit_mask[list(zip(*nav_mask_indices))] = 1
        ask_logit_mask[list(zip(*ask_mask_indices))] = 1

        # Image features
        f_t = self._feature_variable(obs)

        # Budget features
        b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

        return (a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov)

    def rollout(self):
        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Sample random ask positions
        if self.random_ask:
            random_ask_positions = [None] * batch_size
            for i, ob in enumerate(obs):
                random_ask_positions[i] = self.random.sample(
                    range(ob['traj_len']), ob['max_queries'])

        # Index initial command
        seq, seq_mask, seq_lengths = self._make_batch(obs)

        # Roll-out bookkeeping
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'goal_viewpoints': ob['goal_viewpoints'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_ask': [],
            'teacher_ask': [],
            'teacher_ask_reason': [],
            'agent_nav': [],
            'subgoals': []
        } for ob in obs]

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)
        decoder_h = None

        if self.coverage_size is not None:
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                              dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initial actions
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
              self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
              self.ask_actions.index('<start>')

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)
        # Whether the agent reaches its destination in the end
        is_success = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0

        # action_subgoals = [[] for _ in range(batch_size)]
        # n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):
            transition = Transition(batch_size) if not self.is_eval else None   # Preparing training data for DQN

            if not self.is_eval:
                transition.add_filter(ended)            # Filter out episode that has already ended

            dqn_states = self.compute_states(batch_size, obs, queries_unused,
                    (a_t, q_t, decoder_h, ctx, seq_mask, cov))

            if not self.is_eval:
                transition.add_states(dqn_states)

            a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask, b_t, cov = dqn_states      # Unpack

            # Run first forward pass to compute ask logit
            _, _, nav_logit, nav_softmax, ask_logit, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask, budget=b_t, cov=cov)

            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                                              traj, ended, time_step)

            # Ask teacher for next ask action
            ask_target, ask_reason = self.teacher.next_ask(obs)
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            if not self.is_eval and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)

            # Determine next ask action by sampling ask_logit
            q_t = self._sample(ask_logit)

            # Find which agents have asked and prepend subgoals to their current instructions.
            ask_target_list = ask_target.data.tolist()
            q_t_list = q_t.data.tolist()
            has_asked = False
            verbal_subgoals = [None] * batch_size
            edit_types = [None] * batch_size
            for i in range(batch_size):
                if ask_target_list[i] != self.ask_actions.index('<ignore>'):
                    if self.random_ask:
                        q_t_list[i] = time_step in random_ask_positions[i]
                    elif self.ask_first:
                        q_t_list[i] = int(queries_unused[i] > 0)
                    elif self.teacher_ask:
                        q_t_list[i] = ask_target_list[i]
                    elif self.no_ask:
                        q_t_list[i] = 0

                if self._should_ask(ended[i], q_t_list[i]):
                    # Query advisor for subgoal.
                    _, verbal_subgoals[i], edit_types[i] = self.advisor(obs[i], q_t_list[i])
                    # Prepend subgoal to the current instruction
                    self.env.modify_instruction(i, verbal_subgoals[i], edit_types[i])
                    # Reset subgoal step index
                    # n_subgoal_steps[i] = 0
                    # Decrement queries unused
                    queries_unused[i] -= 1
                    # Mark that some agent has asked
                    has_asked = True

            if has_asked:
                # Update observations
                obs = self.env.get_obs()
                # Make new batch with new instructions
                seq, seq_mask, seq_lengths = self._make_batch(obs)
                # Re-encode with new instructions
                ctx, _ = self.model.encode(seq, seq_lengths)
                # Make new coverage vectors
                if self.coverage_size is not None:
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                                      dtype=torch.float, device=self.device)
                else:
                    cov = None

            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
            if not self.is_eval:
                transition.add_actions(q_t)         # Keep track of the ask actions for DQN
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)
            decoder_h, alpha, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                budget=b_t, cov=cov)

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                                              traj, ended, time_step)

            # Ask teacher for next nav action
            nav_target = self.teacher.next_nav(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            # Nav loss
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)
            # Determine next nav action
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)

            # Translate agent action to environment action
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Conditioned on teacher action during intervention
                # (training only or when teacher_interpret flag is on)
                # if (self.teacher_interpret or not self.is_eval) and \
                #         n_subgoal_steps[i] < len(action_subgoals[i]):
                #     a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                #     n_subgoal_steps[i] += 1

                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])

            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)

            # Take nav action
            obs = self.env.step(env_action)

            # Save trajectory output
            ask_target_list = ask_target.data.tolist()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['teacher_ask'].append(ask_target_list[i])
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['teacher_ask_reason'].append(ask_reason[i])
                    traj[i]['subgoals'].append(verbal_subgoals[i])

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                            time_step >= ob['traj_len'] - 1:
                        ended[i] = True

                        if not self.is_eval:
                            # Evaluate whether we ended up in the correct spot
                            instr_id = traj[i]['instr_id']
                            path = traj[i]['agent_path']
                            is_success[i] = self.train_evaluator.score_path(instr_id, path)

                assert queries_unused[i] >= 0

            if not self.is_eval:
                transition.add_is_done(ended)               # Keep track of the episode that are ending for DQN
                transition.add_is_success(is_success)
                transition.compute_reward_shaping()
                dqn_next_states = self.compute_states(batch_size, obs, queries_unused,
                        (a_t, q_t, decoder_h, ctx, seq_mask, cov))
                transition.add_next_states(dqn_next_states)

                experiences = transition.to_list()
                self.buffer.push_multiple(experiences)
                print(f"Just collected {len(experiences)} at time_step {time_step}, buffer size: {len(self.buffer)}!")

                # TODO: Invoke training routine after every interval

            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            self._compute_loss()

        return traj
