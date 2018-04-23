"""
    Experiment script intended to test random covariance matrices
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from tools import statistics, utils
from tools.supervisor import GaussianSupervisor
import argparse
import time as timer
import framework

def main():
    title = 'test_rand'
    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)                         # OpenAI gym environment
    ap.add_argument('--t', required=True, type=int)                     # time horizon
    ap.add_argument('--iters', required=True, type=int, nargs='+')      # iterations to evaluate the learner on
    ap.add_argument('--trace', required=True, type=float)               # trace on the amount of error one expects in the learner
    
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


    def run_iters(self):
        T = self.params['t']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': [],
            'data_used': [],
        }
        trajs = []

        d = self.params['d']
        new_cov = np.random.normal(0, 1, (d, d))
        new_cov = new_cov.T.dot(new_cov)
        new_cov = new_cov / np.trace(new_cov) * self.params['trace']
        self.sup = GaussianSupervisor(self.net_sup, new_cov)

        snapshots = []
        for i in range(self.params['iters'][-1]):
            print "\tIteration: " + str(i)


            states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
            trajs.append((states, i_actions))
            states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
            
            self.lnr.add_data(states, i_actions)

            if ((i + 1) in self.params['iters']):
                snapshots.append((self.lnr.X[:], self.lnr.y[:]))

        for j in range(len(snapshots)):
            X, y = snapshots[j]
            self.lnr.X, self.lnr.y = X, y
            self.lnr.train(verbose=True)
            print "\nData from snapshot: " + str(self.params['iters'][j])
            it_results = self.iteration_evaluation()
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])
            results['data_used'].append(len(y))


        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

