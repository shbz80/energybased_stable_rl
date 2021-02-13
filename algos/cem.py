"""Cross Entropy Method."""
import collections

from dowel import logger, tabular
import numpy as np

from garage import EpisodeBatch, log_performance
from garage.np import paths_to_tensors
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.sampler import RaySampler, LocalSampler
from energybased_stable_rl.utilities.param_exp import sample_params, cem_init_std, cem_stat_compute

class CEM(RLAlgorithm):
    """Cross Entropy Method.

    CEM works by iteratively optimizing a gaussian distribution of policy.

    In each epoch, CEM does the following:
    1. Sample n_samples policies from a gaussian distribution of
       mean cur_mean and std cur_std.
    2. Collect episodes for each policy.
    3. Update cur_mean and cur_std by doing Maximum Likelihood Estimation
       over the n_best top policies in terms of return.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        baseline(garage.np.baselines.Baseline): Baseline for GAE
            (Generalized Advantage Estimation).
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        best_frac (float): The best fraction.
        init_std (float): Initial std for policy param distribution.
        extra_std (float): Decaying std added to param distribution.
        extra_decay_time (float): Epochs that it takes to decay extra std.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 discount=0.99,
                 init_std=1,
                 init_log_std = 0.5,
                 best_frac=0.05,
                 action_lt=10.0,
                 extra_std=0.,
                 extra_decay_time=100):
        self.policy = policy
        self.max_episode_length = env_spec.max_episode_length

        self.sampler_cls = LocalSampler

        self._best_frac = best_frac
        self._baseline = baseline
        self._init_std = init_std
        self._init_log_std = init_log_std
        self._extra_std = extra_std
        self._extra_decay_time = extra_decay_time
        self._episode_reward_mean = collections.deque(maxlen=100)
        self._env_spec = env_spec
        self._discount = discount
        self._n_samples = n_samples

        self._cur_std = {}
        self._cur_mean = None
        self._cur_params = None
        self._all_returns = None
        self._all_params = None
        self._n_best = None
        self._n_params = None
        self._action_lt = action_lt

    def train(self, trainer):
        """Initialize variables and start training.

        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        # epoch-wise
        self._cur_mean = self.policy.get_param_values()
        cem_init_std(self._cur_mean, self._cur_std, self._init_std, self._init_log_std)

        # epoch-cycle-wise
        self._cur_params = self._cur_mean
        self._all_returns = []
        self._all_params = [self._cur_mean.copy()]
        # constant
        self._n_best = int(self._n_samples * self._best_frac)
        assert self._n_best >= 1, (
            'n_samples is too low. Make sure that n_samples * best_frac >= 1')
        # self._n_params = len(self._cur_mean) # todo

        # start actual training
        last_return = None

        for _ in trainer.step_epochs():
            trainer.step_path = []
            for _ in range(self._n_samples):
                action0 = self._action_lt*2.0
                i=0
                while np.any(np.abs(action0) > self._action_lt):
                    self._cur_params = sample_params(self._cur_mean, self._cur_std, trainer.step_itr)
                    self.policy.set_param_values(self._cur_params)
                    self.policy._module.min_icnn()
                    obs = trainer._sampler._workers[0].start_episode()
                    action0, _ = self.policy.get_action(obs)
                    i=i+1
                    print('CEM init trials:',i)
                self._all_params.append(self._cur_params.copy())

                step_path = trainer.obtain_samples(trainer.step_itr)
                last_return = self.train_once(trainer.step_itr,
                                              step_path)
                trainer.step_itr += 1
                trainer.step_path.append(step_path[0])



        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            float: The average return of epoch cycle.

        """
        # -- Stage: Calculate baseline
        print('Train begin')
        if hasattr(self._baseline, 'predict_n'):
            baseline_predictions = self._baseline.predict_n(paths)
        else:
            baseline_predictions = [
                self._baseline.predict(path) for path in paths
            ]

        # -- Stage: Pre-process samples based on collected paths
        samples_data = paths_to_tensors(paths, self.max_episode_length,
                                        baseline_predictions, self._discount)

        # -- Stage: Run and calculate performance of the algorithm
        undiscounted_returns = log_performance(itr,
                                               EpisodeBatch.from_list(
                                                   self._env_spec, paths),
                                               discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))
        samples_data['average_return'] = np.mean(undiscounted_returns)

        epoch = itr // self._n_samples
        i_sample = itr - epoch * self._n_samples
        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)
        # -- Stage: Process samples_data
        rtn = samples_data['average_return']
        self._all_returns.append(samples_data['average_return'])

        # -- Stage: Update policy distribution.
        if (itr + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_returns)
            best_inds = list(np.argsort(-avg_rtns)[:self._n_best])
            best_params = [self._all_params[i] for i in best_inds]

            # MLE of normal distribution
            cem_stat_compute(best_params,self._cur_mean, self._cur_std)
            self.policy.set_param_values(self._cur_mean)

            # Clear for next epoch
            rtn = max(self._all_returns)
            self._all_returns.clear()
            self._all_params.clear()

        # -- Stage: Generate a new policy for next path sampling
        # self._cur_params = sample_params(self._cur_mean, self._cur_std, itr)
        # print(self._cur_params)
        # self._all_params.append(self._cur_params.copy())      #todo
        # self.policy.set_param_values(self._cur_params)

        logger.log(tabular)
        print('Train end')
        return rtn
