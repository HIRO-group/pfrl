import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch.nn import functional as F

import pfrl
from pfrl.agent import (
    GoalConditionedBatchAgent,
    BatchAgent,
    AttributeSavingMixin
)
from pfrl.agents import TD3
from pfrl.utils.batch_states import batch_states
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.replay_buffer import high_level_batch_experiences_with_goal, low_level_batch_experiences_with_goal
from pfrl.replay_buffer import ReplayUpdater
from pfrl.utils import clip_l2_grad_norm_


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)


class GoalConditionedTD3(TD3, GoalConditionedBatchAgent):
    """
    Goal conditioned
    Twin Delayed Deep Deterministic Policy Gradients (TD3).

    See http://arxiv.org/abs/1802.09477

    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        policy_update_delay (int): Delay of policy updates. Policy is updated
            once in `policy_update_delay` times of Q-function updates.
        target_policy_smoothing_func (callable): Callable that takes a batch of
            actions as input and outputs a noisy version of it. It is used for
            target policy smoothing when computing target Q-values.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_policy",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
    )

    def __init__(
        self,
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        n_times_update=1,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        policy_update_delay=2,
        is_low_level=True,
        buffer_freq=10,
        target_policy_smoothing_func=default_target_policy_smoothing_func,
    ):
        # determines if we're dealing with a low level controller.
        self.is_low_level = is_low_level
        self.buffer_freq = buffer_freq
        self.state_arr = []
        self.action_arr = []
        super(GoalConditionedTD3, self).__init__(policy,
                                                 q_func1,
                                                 q_func2,
                                                 policy_optimizer,
                                                 q_func1_optimizer,
                                                 q_func2_optimizer,
                                                 replay_buffer,
                                                 gamma,
                                                 explorer,
                                                 gpu,
                                                 replay_start_size,
                                                 minibatch_size,
                                                 update_interval,
                                                 phi,
                                                 soft_update_tau,
                                                 n_times_update,
                                                 max_grad_norm,
                                                 logger,
                                                 batch_states,
                                                 burnin_action_func,
                                                 policy_update_delay,
                                                 target_policy_smoothing_func)

    def update_q_func_with_goal(self, batch):
        """
        Compute loss for a given Q-function, or critics
        """

        batch_next_state = batch["next_state"]
        batch_next_goal = batch["next_goal"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_goal = batch["goal"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(
            self.target_policy
        ), pfrl.utils.evaluating(self.target_q_func1), pfrl.utils.evaluating(
            self.target_q_func2
        ):
            next_actions = self.target_policy_smoothing_func(
                self.target_policy(torch.cat([batch_next_state, batch_next_goal], -1)).sample()
            )
            next_q1 = self.target_q_func1((torch.cat([batch_next_state, batch_next_goal], -1), next_actions))
            next_q2 = self.target_q_func2((torch.cat([batch_next_state, batch_next_goal], -1), next_actions))
            next_q = torch.min(next_q1, next_q2)

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q)

        predict_q1 = torch.flatten(self.q_func1((torch.cat([batch_state, batch_goal], -1), batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((torch.cat([batch_state, batch_goal], -1), batch_actions)))

        loss1 = F.mse_loss(target_q, predict_q1)
        loss2 = F.mse_loss(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        self.q_func1_optimizer.zero_grad()
        loss1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        self.q_func2_optimizer.zero_grad()
        loss2.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()

        self.q_func_n_updates += 1

    def update_policy_with_goal(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]
        batch_goal = batch["goal"]

        onpolicy_actions = self.policy(torch.cat([batch_state, batch_goal], -1)).rsample()
        q = self.q_func1((torch.cat([batch_state, batch_goal], -1), onpolicy_actions))

        # Since we want to maximize Q, loss is negation of Q
        loss = -torch.mean(q)

        self.policy_loss_record.append(float(loss))
        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.policy_n_updates += 1

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        if self.is_low_level:
            batch = low_level_batch_experiences_with_goal(experiences, self.device, self.phi, self.gamma)
            self.update_q_func_with_goal(batch)
            if self.q_func_n_updates % self.policy_update_delay == 0:
                self.update_policy_with_goal(batch)
                self.sync_target_network()
        else:
            # dealing with high level controller
            batch = high_level_batch_experiences_with_goal(experiences, self.device, self.phi, self.gamma)


    def batch_select_onpolicy_action(self, batch_obs):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            batch_action = self.policy(batch_xs).sample().cpu().numpy()
        return list(batch_action)

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_act_with_goal(self, batch_obs, batch_goal):
        if self.training:
            return self._batch_act_train_goal(batch_obs, batch_goal)
        else:
            return self._batch_act_eval_goal(batch_obs)

    def batch_observe_with_goal(self, batch_obs, batch_goal, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train_goal(batch_obs, batch_goal, batch_reward, batch_done, batch_reset)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval_goal(self, batch_obs, batch_goal):
        assert not self.training
        concat_states = []
        for idx, ob in enumerate(batch_obs):
            concat_states.append(torch.cat([ob, batch_goal[idx]]))
        return self.batch_select_onpolicy_action(concat_states)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self.batch_select_onpolicy_action(batch_obs)

    def _batch_act_train_goal(self, batch_obs, batch_goal):
        assert self.training
        if self.burnin_action_func is not None and self.policy_n_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            concat_states = []
            for idx, ob in enumerate(batch_obs):
                concat_states.append(torch.cat([ob, batch_goal[idx]]))
            batch_onpolicy_action = self.batch_select_onpolicy_action(concat_states)
            batch_action = [
                self.explorer.select_action(self.t, lambda: batch_onpolicy_action[i])
                for i in range(len(batch_onpolicy_action))
            ]

        self.batch_last_obs = list(batch_obs)
        self.batch_last_goal = list(batch_goal)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_act_train(self, batch_obs):
        assert self.training
        if self.burnin_action_func is not None and self.policy_n_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            batch_onpolicy_action = self.batch_select_onpolicy_action(batch_obs)
            batch_action = [
                self.explorer.select_action(self.t, lambda: batch_onpolicy_action[i])
                for i in range(len(batch_onpolicy_action))
            ]

        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_train_goal(self, batch_obs, batch_goal, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_goal[i] is not None
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                if self.is_low_level:
                    self.replay_buffer.append(
                        state=self.batch_last_obs[i],
                        goal=self.batch_last_goal[i],
                        action=self.batch_last_action[i],
                        reward=batch_reward[i],
                        next_state=batch_obs[i],
                        next_goal=batch_goal[i],
                        next_action=None,
                        is_state_terminal=batch_done[i],
                        env_id=i,
                    )
                else:
                    if self.t % self.buffer_freq == 1:
                        self.replay_buffer.append(
                            state=self.batch_last_obs[i],
                            goal=self.batch_last_goal[i],
                            action=self.batch_last_action[i],
                            reward=batch_reward[i],
                            next_state=batch_obs[i],
                            next_action=None,
                            is_state_terminal=batch_done[i],
                            state_arr=self.state_arr,
                            action_arr=self.action_arr,
                            env_id=i
                        )
                        self.state_arr = []
                        self.action_arr = []
                    self.state_arr.append(self.batch_last_obs[i])
                    self.action_arr.append(self.batch_last_action[i])

                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_goal[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)


    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

    def get_statistics(self):
        return [
            ("average_q1", _mean_or_nan(self.q1_record)),
            ("average_q2", _mean_or_nan(self.q2_record)),
            ("average_q_func1_loss", _mean_or_nan(self.q_func1_loss_record)),
            ("average_q_func2_loss", _mean_or_nan(self.q_func2_loss_record)),
            ("average_policy_loss", _mean_or_nan(self.policy_loss_record)),
            ("policy_n_updates", self.policy_n_updates),
            ("q_func_n_updates", self.q_func_n_updates),
        ]