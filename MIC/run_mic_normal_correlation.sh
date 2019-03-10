#!/usr/bin/env bash

X="../output/normal_otu.txt"
ODIR="normal_results"
rm -rf $ODIR
mkdir $ODIR

source activate DL_2018_1_23

mictools null $X $ODIR/null_dist.txt

mictools pval $X $ODIR/null_dist.txt $ODIR

mictools adjust $ODIR/pval.txt $ODIR

mictools strength $X $ODIR/pval_adj.txt $ODIR/strength.txt

source deactivate DL_2018_1_23