#!/bin/sh

# loop over all possible weights of 1ake.pdb
for i in $(seq 0 0.1 1); do
python ~/git/cryoBioEN/cryoBioEN.py 4ake_10A.mrc $i 10 0.1
done
