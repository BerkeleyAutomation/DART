import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20", 'purple'))
import IPython


def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--iters', required=True, type=int, nargs='+')
    ap.add_argument('--update', required=True, nargs='+', type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--normalize', action='store_true', default=False)
    
    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 50

    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    plt.style.use('ggplot')

    iters = params['iters']
    ptype = 'data_used'
    


    # BC
    bc_data = []

    title = 'test_bc'
    ptype = 'data_used'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning
    c = next(color)
    try:
        means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
        bc_data.append(means)
    except IOError:
        pass


    bc_data = np.array(bc_data)
    bc_data = np.sum(bc_data[:, -1])

    # DAgger B
    betas = [.1, .3, .5, .7, .9]
    colors = ['blue', 'red', 'black', 'pink', 'aqua']
    dagger_b_data = []
    for beta, c in zip(betas, colors):

        title = 'test_dagger_b_beta' + str(beta)
        ptype = 'data_used'
        params_dagger_b = params.copy()
        params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
        # c = next(color)
        try:
            means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
            dagger_b_data.append(means)
        except IOError:
            pass

    dagger_b_data = np.array(dagger_b_data)
    dagger_b_data = np.sum(dagger_b_data[:, -1])


    betas = [.1, .3, .5, .7, .9]
    colors = ['blue', 'red', 'black', 'pink', 'aqua']
    dagger_b_data2 = []
    for beta, c in zip(betas, colors):

        title = 'test_dagger_b2_beta' + str(beta)
        ptype = 'data_used'
        params_dagger_b = params.copy()
        params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
        # c = next(color)
        try:
            means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
            dagger_b_data2.append(means)
        except IOError:
            pass

    IPython.embed()
    dagger_b_data2 = np.array(dagger_b_data2)
    dagger_b_data2 = np.sum(dagger_b_data2[:, -1])


    labels = ['BC', 'Dagger-b', 'Dagger-b2']
    data = [bc_data, dagger_b_data, dagger_b_data2]
    plt.bar(labels, data)

    IPython.embed()

    # betas = [.5]
    # colors = ['blue', 'red', 'black', 'pink', 'aqua']
    # for beta, c in zip(betas, colors):

    #     title = 'test_dagger_b2_beta' + str(beta)
    #     ptype = 'reward'
    #     params_dagger_b = params.copy()
    #     params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
    #     # c = next(color)
    #     try:
    #         means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
    #         means, sems = normalize(means, sems)
    #         plt.plot(iters, means, color=c, label=title)
    #         plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    #     except IOError:
    #         pass
      


    # # DART
    # title = 'test_dart'
    # ptype = 'reward'
    # params_dart = params.copy()
    # c = next(color)
    # try: 
    #     means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
    #     means, sems = normalize(means, sems)
    #     plt.plot(iters, means, label='DART', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass


    # plt.title("Reward on " + str(params['envname']))
    # plt.legend()
    # plt.xticks(iters)
    # plt.legend()
    # if should_normalize:
    #     plt.ylim(0, 1.05)
    #     plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # save_path = 'images/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # if should_save == True:
    #     plt.savefig(save_path + str(params['envname']) + "_reward.pdf")
    # else:
    #     plt.show()



if __name__ == '__main__':
    main()


