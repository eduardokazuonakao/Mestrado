import numpy as np
import scipy as sp
import scipy.stats as st
import itertools as it

def friedman_test(*args):
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')
    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])
    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r/sp.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]
    chi2 = ((12*n)/float((k*(k+1))))*((sp.sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))
    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))
    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))
    return iman_davenport, p_value, rankings_avg, rankings_cmp

def bonferroni_dunn_test(ranks, control=None):
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())
    versus = list(it.combinations(range(k), 2))
    comparisons = [keys[vs[0]] + " vs " + keys[vs[1]] for vs in versus]
    z_values = [abs(values[vs[0]] - values[vs[1]]) for vs in versus]
    p_values = [2*(1-st.norm.cdf(abs(z))) for z in z_values]
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min((k-1)*p_value,1) for p_value in p_values]
    return comparisons, z_values, p_values, adj_p_values


NDR = np.array([-0.430, 0.463, 0.872, 0.953, 0.800])
print('Média NDR:', NDR.mean())

PCA = np.array([-0.303, 0.460, 0.899, 0.960, 0.851])
print('Média PCA:', PCA.mean())

ISOMAP = np.array([-0.408, 0.467, 0.875, 0.954, 0.787])
print('Média ISOMAP:', ISOMAP.mean())

LLE = np.array([-0.285, 0.459, 0.909, 0.959, 0.850])
print('Média LLE:', LLE.mean())
print()

F, pvalue, rankings, pivots = friedman_test(NDR, PCA, ISOMAP, LLE)
print('***** Friedman results')
print('p-value = %.20f' %pvalue)
print()

pivots_dict = dict([('NDR', pivots[0]), ('PCA', pivots[1]), ('ISOMAP', pivots[2]), ('LLE', pivots[3])])

comparisonsB, zvaluesB, pvaluesB, adjustedpvaluesB = bonferroni_dunn_test(pivots_dict)
print('***** Post-hoc Bonferroni-Dunn')
print('Comparações: ', comparisonsB)
print('p-values: ', pvaluesB)
print()