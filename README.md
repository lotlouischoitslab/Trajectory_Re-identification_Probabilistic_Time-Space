# Autonomous Vehicles Trajectory Prediction Analysis

## Contributors:
- **Louis Sungwoo Cho, Civil & Environmental Engineering (Transportation) Major, Computer Science Minor, </br> University of Illinois at Urbana-Champaign (UIUC)**
- **Alireza Talebpour, Assistant Professor Civil & Environmental Engineering (Transportation), </br> University of Illinois at Urbana-Champaign (UIUC)**


## HAL Cluster Notes:

File Name: predict_environment_works_with_six_maneuvers_model_10_sec.py

## NOTES For Louis Sungwoo Cho NCAS HAL Cluster:
**[Reference Video Link](https://www.youtube.com/watch?v=l1dV25xwo0o&list=PLO8UWE9gZTlCtkZbWtEcKgxYVVLIvN2IS&index=1)**

Run the GPU: 

         swrun -p gpux1 -r louissc2
Exit the terminal:

         exit 
         
We are using CEE497 conda environment and go here for more reference: https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster 
Make sure to upload the files to the cluster if you have made any changes

We need to: 

         conda install -c "conda-forge/label/cf202003" libopenblas
         
To connect to NCSA Hal Cluster: 

         conda config --add channels https://ftp.osuosl.org/pub/open-ce/1.5.1/
         
Type in Password & Enter the Authentication code:

         module load opence
         conda activate CEE497
         
To save: If you're using vim, you can press ESC, then type :wq and press Enter.

        ./demo.swb
        Type the following:
        #!/bin/bash 
        #SBATCH --job-name="louis_trajectory"
        #SBATCH --output="louis_trajectory.out"
        #SBATCH --partition=gpux1
        #SBATCH --time=2
        #SBATCH --reservation=louissc2

        module load wmlce

        hostname 

        ./demo2.swb
        Type the following: 
        #!/bin/bash 
        #SBATCH --job-name="louis_trajectory"
        #SBATCH --output="louis_trajectory.out"
        #SBATCH --partition=gpu
        #SBATCH --time=2

        module load wmlce

        hostname 


        Run the batch 
        swbatch ./demo.swb

        Check Status 
        squeue -u louissc2

        Launch VIM
        vim ./demo.s
        Quit: :q!
## Run GPU on HAL Cluster:
        swqueue (GPUs and the queue of users)
        squeue (List of currently running clusters)   
        sinfo
        swrun -p gpux1 
        module load wmlce
 
