#!/bin/bash

# Job name:
#SBATCH --job-name=train
#
# Project:
#SBATCH --account=ec-marcuber
#
# Wall time limit:
#SBATCH --time=DD-03:00:00
#

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load main.py
module list

## Do some work:
python main.py