# ImageClassification-MultilayerPerceptrons-CNNs

## SLURM x MIMI
To load the Slurm module and submit a job, use the following commands:
```bash
module load slurm 
sbatch launch.sh
```

To see what's happening with all jobs:
```bash
squeue
```

To see the status of your jobs:
```bash
squeue -u #USER
```

To display the output of the GPU:
```bash
cat gpu-grad-01-63.out
```
Replace `gpu-grad-01-63` with the name of the computer.