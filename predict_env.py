import numpy as np


class PredictEnv:
    def __init__(self, model, env_name, model_type, args):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.args = args

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2


            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "HalfCheetah-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Ant-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * (x >= 0.2) \
                       * (x <= 1.0)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Humanoid-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:, 0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:, None]
            return done
        elif env_name == "Swimmer-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Striker-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Thrower-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Pusher-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "Reacher-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done
        elif env_name == "InvertedPendulum-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            notdone = np.isfinite(next_obs).all(axis=-1) \
                      * (np.abs(next_obs[:, 1]) <= .2)
            done = ~notdone
            done = done[:, None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds


    def random_inds(self, batch_size):
        inds = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        return inds

    def max_inds(self, weight, batch_size):
        inds = np.argmax(weight)
        inds = np.repeat(inds, repeats=batch_size, axis=0)
        return inds

    def greedy(self, e, weight, batch_size):
        inds = []
        n = 0
        for i in range(batch_size):
            if np.random.random() <= e:
                print(np.argmax(weight))
                inds.append(np.argmax(weight))
            else:
                inds.append(np.random.choice(self.model.sorted_loss_idx))
            if n >= batch_size - 1:
                break
            n += 1
        inds = np.array(inds)
        return inds

    def weighted_random_inds(self, weight, batch_size):
        # p = weight[self._model_inds]
        p = np.array([weight[i] for i in self.model.sorted_loss_idx])
        p = 1/p
        p = p/np.sum(p)
        inds = np.random.choice(self.model.sorted_loss_idx, p=p.ravel(), size=batch_size)
        return inds


    def step(self, obs, act, deterministic=True):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)

        # uncertainty = np.std(ensemble_model_means, axis=0)

        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        # if self.args.model_select_method == 'weighted':
        #     model_idxes = self.weighted_random_inds(self.model.holdout_mse_losses, batch_size)
        # elif self.args.model_select_method == 'random':
        model_idxes = self.random_inds(batch_size)
        # else:
        #     raise ValueError('model selection type error')

        batch_idxes = np.arange(0, batch_size)

        reward_var = np.var(ensemble_samples[:, :, :1], 0)
        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        rewards -= reward_var
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info
