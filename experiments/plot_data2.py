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
    params['epochs'] = 100

    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    plt.style.use('ggplot')

    iters = params['iters']
    ptype = 'data_used'
    

    # DAgger B
    betas = [.1, .3, .5, .7, .9]
    colors = ['blue', 'red', 'black', 'pink', 'aqua']
    dagger_b_data = []
    dagger_b_sems = []
    for beta, c in zip(betas, colors):

        title = 'test_dagger_b'
        ptype = 'data_used'
        params_dagger_b = params.copy()
        params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
        try:
            means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
            dagger_b_data.append(means)
            dagger_b_sems.append(sems)
        except IOError:
            pass

    dagger_b_data = np.array(dagger_b_data)
    dagger_b_sems = np.array(dagger_b_sems)
    dagger_b_data = np.sum(dagger_b_data[:, -1])
    sems = dagger_b_sems[:, -1]
    dagger_b_sem = np.sqrt(np.sum(sems ** 2.0))



    parts = [10]
    dart_names = ['DART ' + str(part) for part in parts]
    dart_data = []
    dart_sem = []
    for part in parts:
        title = 'test_dart'
        ptype = 'data_used'
        params_dart = params.copy()
        params_dart['partition'] = part
        try: 
            means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
            dart_data.append(means[-1])
            dart_sem.append(sems[-1])

        except IOError:
            pass




    labels = ['Dagger-B']
    data = [dagger_b_data]
    errs = [dagger_b_sem]
    labels = labels + dart_names
    data = data + dart_data
    errs = errs + dart_sem
    plt.bar(labels, data, yerr=errs)
    plt.title(params['envname'][:-3])

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_data2.pdf")
        plt.savefig(save_path + "svg_" + str(params['envname']) + "_data2.svg")
    else:
        plt.show()



if __name__ == '__main__':
    main()


