# ImageClassification-MultilayerPerceptrons-CNNs

## SLURM x MIMI
You must first go to the `Output` Folder
```bash
cd ImageClassification-MultilayerPeceptron-CNNs/Output
```

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

## Clearing *.out files on mimi
On mimi, you may end up with many *.out files. To clear them, first go to the `Output` Folder:
```bash
cd ImageClassification-MultilayerPeceptron-CNNs/Output
```
Then run the following command:
```bash
python3 clear-output-files.py
```