#! /bin/bash
#
# Copyright (C) 2015  <>
#
# Distributed under terms of the MIT license.
#

DATASETS=("baudio" "jester" "bnetflix" "accidents" "tmovie" \
	  "kdd" "tretail" "msweb" "kosarek" "pumsb_star" "plants" \
	  "nltcs" "dna" "msnbc" "book" "cwebkb" "cr52" "c20ng" "bbc" "ad")
for DATASET in "${DATASETS[@]}"
do
    qsub -pe orte 4 -l gpu=0 -l virtual_free=4G -l h_rt=144:00:00 -o "./onlines/$DATASET.cvb.out" \
        -e "./onlines/$DATASET.cvb.err" -v DATASET=$DATASET online_cvb.sh
done
