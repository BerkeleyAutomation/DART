"""
    Experiment script intended to test DART
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from tools import statistics, noise, utils
from tools.supervisor import GaussianSupervisor
import argparse
import scipy.stats
import time as timer
import framework

def main():
    title = 'test_dart'
    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)                         # OpenAI gym environment
    ap.add_argument('--t', required=True, type=int)                     # time horizon
    ap.add_argument('--iters', required=True, type=int, nargs='+')      # iterations to evaluate the learner on
    ap.add_argument('--update', required=True, nargs='+', type=int)     # iterations to update the noise term
    ap.add_argument('--partition', required=True, type=int)             # Integer between 1 and 450 (exclusive),

    args = vars(ap.parse_args())
    args['arch'] = [64, 64]
    args['lr'] = .01
    args['epochs'] = 100

    TRIALS = framework.TRIALS


    test = Test(args)
    start_time = timer.time()
    test.run_trials(title, TRIALS)
    end_time = timer.time()

    print "\n\n\nTotal time: " + str(end_time - start_time) + '\n\n'



class Test(framework.Test):

    def count_states(self, trajs):
        count = 0
        for states, actions in trajs:
            count += len(states)
        return count



    def update_noise(self, i, trajs):

        if i in self.params['update']:
            self.optimized_data = self.count_states(trajs)
            self.lnr.train()
            new_cov = noise.sample_covariance_trajs(self.env, self.lnr, trajs, 5, self.params['t'])
            new_cov = new_cov
            print "Estimated covariance matrix: "
            print new_cov
            print np.trace(new_cov)
            self.sup = GaussianSupervisor(self.net_sup, new_cov)
            return self.sup
        else:
            return self.sup



    def run_iters(self):
        T = self.params['t']
        partition = self.params['partition']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': [],
            'data_used': [],
        }
        trajs = []
        snapshots = []
        traj_snapshots = []
        self.optimized_data = 0

        for i in range(self.params['iters'][-1]):
            print "\tIteration: " + str(i)

            self.sup = self.update_noise(i, trajs)

            states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
            states, i_actions, (held_out_states, held_out_actions) = utils.filter_data(self.params, states, i_actions)

            rang = np.arange(0, len(held_out_states))
            np.random.shuffle(rang)
            noise_states, noise_actions = [held_out_states[k] for k in rang[:partition]], [held_out_actions[k] for k in rang[:partition]]

            trajs.append((noise_states, noise_actions))
            self.lnr.add_data(states, i_actions)

            if ((i + 1) in self.params['iters']):
                snapshots.append((self.lnr.X[:], self.lnr.y[:]))
                traj_snapshots.append(self.optimized_data)

        for j in range(len(snapshots)):
            X, y = snapshots[j]
            optimized_data = traj_snapshots[j]
            self.lnr.X, self.lnr.y = X, y
            self.lnr.train(verbose=True)
            print "\nData from snapshot: " + str(self.params['iters'][j])
            it_results = self.iteration_evaluation()
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])
            results['data_used'].append(len(y) + optimized_data)
            print "\nTrain data: " + str(len(y))
            print "\n Optimize data: " + str(optimized_data)

        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

