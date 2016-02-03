import os, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def parse(filename, max_iter=30):
    iters, train_logps, valid_logps = [], [], []
    with file(filename, 'r') as fin:
        for line in fin:
            segments = line.split(',')
            if len(segments) == 3:
                iters.append(int(segments[0]))
                train_logps.append(float(segments[1]))
                valid_logps.append(float(segments[2]))
    return np.asarray(iters)[:max_iter]-min(iters), -np.asarray(train_logps)[:max_iter], np.asarray(valid_logps)[:max_iter]


datasets = {'accidents':'Accidents', 'ad':'Ad', 'baudio':'Audio', 'bbc':'BBC', 'bnetflix':'Netflix',
            'book':'Book', 'c20ng':'20 Newsgroup', 'cr52':'Reuters-52', 'cwebkb':'WebKB', 'dna':'DNA',
            'jester':'Jester', 'kdd':'KDD 2000', 'kosarek':'Kosarek', 'msnbc':'MSNBC', 'msweb':'MSWeb', 
            'nltcs':'NLTCS', 'plants':'Plants', 'pumsb_star':'Pumsb Star', 'tmovie':'EachMovie', 'tretail':'Retail'}

#algos = ['pgd', 'eg', 'sma', 'cccp', 'vbem']
algos = ['pgd', 'vbem']

title = sys.argv[1]

pgd_iters, pgd_train_logps, pgd_valid_logps = parse('./{}.pgd.out'.format(title))
#eg_iters, eg_train_logps, eg_valid_logps = parse('./{}.eg.out'.format(title))
#sma_iters, sma_train_logps, sma_valid_logps = parse('./{}.sma.out'.format(title))
#em_iters, em_train_logps, em_valid_logps = parse('./{}.em.out'.format(title))
vbem_iters, vbem_train_logps, vbem_valid_logps = parse('./{}.vbem.out'.format(title))

plt.figure()
plt.plot(pgd_iters, pgd_train_logps, 'rx-', linewidth=2.0, markersize=8)
#plt.plot(eg_iters, eg_train_logps, 'bs-', linewidth=2.0, markersize=8)
#plt.plot(sma_iters, sma_train_logps, 'g^-', linewidth=2.0, markersize=8)
#plt.plot(em_iters, em_train_logps, 'mo-', linewidth=2.0, markersize=8)
plt.plot(vbem_iters, vbem_train_logps, 'bs-', linewidth=2.0, markersize=8)
plt.grid(True)
plt.title(datasets[title], fontsize=20, fontweight='bold')
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('$-\log\Pr(\mathbf{x}|\mathbf{w})$', fontsize=20)
#plt.legend(['PGD', 'EG', 'SMA', 'CCCP'])
plt.legend(['PGD', 'VBEM'])
plt.savefig('{}-copts.pdf'.format(title), bbox_inches='tight')
plt.show()
