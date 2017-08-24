import pickle
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib as mpl
import numpy as np
import sys, argparse
import os

from Semibandits2 import Experiment

Names = {
    'mini_gb2': 'MM-GB2',
    'mini_gb5': 'MM-GB5',   
    'mini_lin': 'MM-Lin',
    'epsall_gb2': '$\epsilon$-GB2',
    'epsall_gb5': '$\epsilon$-GB5',
    'epsall_lin': '$\epsilon$-Lin',
    'lin': 'LinUCB',
    'rucb_lin': 'RUCB-Lin',
    'rucb_tree': 'RUCB-Tree',
    'rucb_gb5': 'RUCB-GB5',
    'rucb_gb5fast': 'RUCB-GB5-Fast'
}

Styles = {
    'mini_gb2': ['k', 'solid'],
    'mini_gb5': ['r', 'solid'],   
    'mini_lin': ['g', 'solid'],
    'epsall_gb2': ['k', 'dashed'],
    'epsall_gb5': ['y', 'solid'],
    'epsall_lin': ['g', 'dashed'],
    'lin': ['b', 'solid'],
    'rucb_lin': ['g', 'solid'],
    'rucb_tree': ['y', 'solid'],
    'rucb_gb5': ['k', 'solid'],
    'rucb_gb5fast': ['m', 'solid'],
    }

def read_dir(path):

    files = os.listdir(path)

    experiments = []

    for fpath in files:

        full_path = path + fpath
        
        try:
            exp = Experiment.load(full_path)
        except RuntimeError:
            print("Couldn't load {}".format(full_path))
            continue

        experiments.append(exp)

    return experiments

def make_string(keys, vals):

    st = None

    for (key, val) in zip(keys, vals):

        if st is None:
            st = "{}_{}".format(key, val)
        else:
            st = st + "_{}_{}".format(key, val)

    return st
    

def filter_experiments(experiments, ftr):

    groups = {}

    for exp in experiments:
        
        attrs = []

        for (idx, key) in enumerate(ftr):

            val = exp.__dict__[key]

            attrs.append(val)

        name = exp.name

        st = make_string(ftr, attrs)

        if st in groups:
            groups[st][name] = exp
        else:
            groups[st] = {name: exp}

    return groups


if __name__ == '__main__':

    DATA_DIR="./results/temp/"
    FILTER = ("alg_name", "learning_alg")
    DPI = 200

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', dest='save', action='store_true')
    Args = parser.parse_args(sys.argv[1:])

    # Map from files to experiments.
    # D1 = read_dir("./results/mslr30k_T=30000_L=1_e=0.0/")
    experiments = read_dir(DATA_DIR)
    
    D1 = filter_experiments(experiments, FILTER)
    print(D1)

    print(mpl.rcParams['figure.figsize'])
    fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0]*2, mpl.rcParams['figure.figsize'][1]*2))
    # fig = plt.figure()
    ax = fig.add_subplot(111,frameon=False)
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)


    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    std = True
    legendHandles = []
    keys = ['epsall_lin', 'mini_lin', 'epsall_gb2', 'mini_gb2', 'epsall_gb5', 'mini_gb5', 'lin', 'rucb_lin', 'rucb_tree', 'rucb_gb5', 'rucb_gb5fast']

    for k in D1.keys():
        mus = []
        stds = []
        # for (k1,v1) in D1[0].items():
        for(k1, v1) in D1[k].items():
            # if k1.find(k) == 0 and len(D1[0][k1]) != 0:

            # Compute running average
            rewards_cum = v1.rewards
            
            x = np.arange(100, 10*len(rewards_cum)+1, 100)
            
            mus.append(rewards_cum[9::10]/x)
            # stds.append(2/np.sqrt(len(D1[0][k1]))*(np.std(D1[0][k1],axis=0)[9::10]/x))
        if len(mus) == 0:
            continue
        A = np.vstack(mus)
        # B = np.vstack(stds)
        # ids = np.argmax(A, axis=0)
        # mu = np.array([A[ids[i], i] for i in range(len(ids))])
        # stdev = np.array([B[ids[i], i] for i in range(len(ids))])

        # if k == 'mini_gb5':
        #     mu = np.mean(D1[0]['mini_gb5_0.010'], axis=0)[9::10]/x
        #     stdev = 2/np.sqrt(len(D1[0]['mini_gb5_0.010']))*(np.std(D1[0]['mini_gb5_0.010'], axis=0)[9::10]/x)
        # if k == 'rucb_gb5':
        #     mu = np.mean(D1[0]['rucb_gb5_0.010'], axis=0)[9::10]/x
        #     stdev = 2/np.sqrt(len(D1[0]['rucb_gb5_0.010']))*(np.std(D1[0]['rucb_gb5_0.010'], axis=0)[9::10]/x)

        # TODO: Fix to not search for k index.

        rep = mus[0]

        name = k
        l1 = ax1.plot(x,rep,rasterized=True, linewidth=2.0, label=name)
        legendHandles.append((matplotlib.patches.Patch(color=l1[0].get_color(), label=name), name))
        # if std and k=='mini_gb5' or k=='lin' or 'rucb_gb5':
        #     ax1.fill_between(x,
        #                      mu - stdev,
        #                      mu + stdev,
        #                      color = l1[0].get_color(), alpha=0.2, rasterized = True)

    # for k in keys:
    #     params = []
    #     mus = []
    #     stds = []
    #     for (k1,v1) in D2[0].items():
    #         if k1.find(k) == 0 and len(D2[0][k1]) != 0:
    #             x = np.arange(100, 10*len(D2[0][k1][0])+1, 100)
    #             mus.append(np.mean(D2[0][k1],axis=0)[9::10]/x)
    #             stds.append(2/np.sqrt(len(D2[0][k1]))*(np.std(D2[0][k1],axis=0)[9::10]/x))
    #         params.append(k1.split("_")[-1])
    #     if len(mus) == 0:
    #         continue
    #     A = np.vstack(mus)
    #     B = np.vstack(stds)
    #     ids = np.argmax(A, axis=0)
    #     mu = np.array([A[ids[i], i] for i in range(len(ids))])
    #     stdev = np.array([B[ids[i], i] for i in range(len(ids))])

    #     # TODO update for RUCB stdev

    #     if k == 'mini_gb5':
    #         mu = np.mean(D2[0]['mini_gb5_0.010'], axis=0)[9::10]/x
    #         stdev = 2/np.sqrt(len(D2[0]['mini_gb5_0.010']))*(np.std(D2[0]['mini_gb5_0.010'], axis=0)[9::10]/x)

    #     l1 = ax2.plot(x,mu,rasterized=True, linewidth=2.0, label=Names[k], color=Styles[k][0], linestyle=Styles[k][1])
    #     if std and k=='mini_gb5' or k=='lin':
    #         ax2.fill_between(x,
    #                          mu - stdev,
    #                          mu + stdev,
    #                          color = l1[0].get_color(), alpha=0.2, rasterized = True)

    plt.rc('font', size=18)
    plt.rcParams['text.usetex'] = True
    plt.rc('font', family='sans-serif')

    ## Ax1 is MSLR
    ticks=ax1.get_yticks()
    print(ticks)
    #ax1.set_ylim(2.15, 2.35)
    print("Setting ylim to %0.2f, %0.2f" % (ticks[3], ticks[len(ticks)-2]))
    ticks = ax1.get_yticks()
    print(ticks)
    ticks = ["", "", "2.2", "", "2.3", ""]
    #ax1.set_yticklabels(ticks,size=20)
    ticks = ['', '', '10000', '', '20000', '', '30000']
    #ax1.set_xlim(1000, 31000)
    #ax1.set_xticklabels(ticks,size=20)

    # Ax2 is Yahoo!
    # ticks=ax2.get_yticks()
    # print(ticks)
    # ax2.set_ylim(2.90,3.12)
    # print("Setting ylim to %0.2f, %0.2f" % (ticks[3], 3.15))
    # ticks=ax2.get_yticks()
    # print(ticks)
    # ticks = ["", "2.9", "", "3.0", "", "3.1"]
    # ax2.set_yticklabels(ticks,size=20)
    # ticks = ['', '', '10000', '', '20000', '', '30000']
    # ax2.set_xlim(1000, 32000)
    # ax2.set_xticklabels(ticks,size=20)

    plt.sca(ax)
    plt.ylabel('Average reward')

    plt.xlabel('Number of interactions (n)')
    # leg = ax1.legend([x[1] for x in legendHandles], loc='upper center', bbox_to_anchor=(-0.1, -0.15), fancybox=False, shadow=False, ncol=7, frameon=False,fontsize=18)
    # leg = ax1.legend([x[1] for x in legendHandles], loc='lower center', bbox_to_anchor=(0.5,-.2), fancybox=False, shadow=False, ncol=7, frameon=False,fontsize=18)
    # leg = ax1.legend([x[1] for x in legendHandles], loc='lower center', fancybox=False, shadow=False, ncol=7, frameon=False,fontsize=18)
    leg = ax1.legend([x[1] for x in legendHandles], loc='lower right')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    plt.sca(ax1)
    tt1 = plt.title('Dataset: MSLR30k',fontsize=18)
    tt1.set_position([0.5, 1.02])
    # plt.sca(ax2)
    # tt2 = plt.title('Dataset: Yahoo!',fontsize=18)
    # tt2.set_position([0.5, 1.02])
    plt.gcf().subplots_adjust(bottom=0.25)
    if Args.save:
        plt.savefig("./figs/plots_mslr30k.png", format='png', dpi=DPI, bbox_inches='tight')
        plt.savefig("./figs/plots_mslr30k.pdf", format='pdf', dpi=DPI, bbox_inches='tight')
    else:
        plt.show()


    ## (DONE) No band
    ## (DONE) markers + update legend
    ## (DONE) No legend frame
    ## (DONE) font is too big
    ## space between title and plot
    ## space between ylabel and yticks

    ## Get P-values (paired ttest and regular ttest)
