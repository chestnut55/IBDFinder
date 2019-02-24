#!/bin/bash
date
# use sparcc to calculate correlation
itr=20 # default 20
num_boot=100
corr_threshold=0.1 # default 0.1

# Downstream cor_to_network.py cutoffs
cor_cut_off=0.3 # min correlation threshold
pval_cut_off=0.05 # max pvalue
### A. Correlation Calculation ###
# 1.calculate disease group correlation
python SparCC.py output/ibd_otus.csv -i $itr --cor_file=test/cor_sparcc.out -t $corr_threshold

### B. Pseudo p-value Calculation ###
#
python MakeBootstraps.py output/ibd_otus.csv -n $num_boot -t permutation_#.txt -p test/pvals/

# compute sparcc on resampled (with replacement) datasets
#1. bootstrap procedure
for ((i=0;i<$num_boot;i++))
do
    python SparCC.py test/pvals/permutation_$i.txt -i $itr --cor_file=test/pvals/perm_cor_$i.txt
done

# 2. compute one sided p-values
python PseudoPvals.py test/cor_sparcc.out test/pvals/perm_cor_#.txt $num_boot -o test/pvals/pvals.one_sided.txt -t one_sided

python cor_to_network.py -i test/cor_sparcc.out --pval test/pvals/pvals.one_sided.txt --cor_cutoff $cor_cut_off --pval_cutoff $pval_cut_off
