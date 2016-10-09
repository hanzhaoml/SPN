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
   DATASET=$DATASET online_adf.sh
done
