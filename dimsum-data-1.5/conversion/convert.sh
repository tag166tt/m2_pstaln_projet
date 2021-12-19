#!/bin/bash
OUT=../dimsum16.train

python3 sst_to_dimsum.py original/lowlands.UPOS2.tsv lowlands > $OUT
python3 sst_to_dimsum.py original/ritter.UPOS2.tsv ritter >> $OUT
python3 streusle_to_dimsum.py original/streusle.upos.tags >> $OUT
