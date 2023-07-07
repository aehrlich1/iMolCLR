#!/bin/bash

# Accepts 1 parameter: finetune.py OR molclr.py
# 1. ssh into titan directory: /home/mescalin/aehrlich/src/iMolCLR
# 2. git pull
# 3. execute submit_job.slrm: sbatch submit_job.slrm

ssh aehrlich@titan.tbi.univie.ac.at 'cd src/iMolCLR; git pull; sbatch ./submission/submit_finetune.slrm'