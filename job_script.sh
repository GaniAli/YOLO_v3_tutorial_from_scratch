#!/bin/bash
#
## otherwise the default shell would be used
#$ -S /bin/bash

## Pass environment variables of workstation to GPU node 
#$ -V

## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue
#$ -q short.q@*

## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=4G

## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferrably on your scratch
#$ -o /path/to/log/dir
#
## send mail on job's end and abort
#$ -m a

# if you need to export custom libs, you can do that here
#export LD_LIBRARY_PATH=/scratch_net/yourhost/yourname/lib/opencv/lib:$LD_LIBRARY_PATH

# call your calculation executable, redirect output

