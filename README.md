## Trajectory Re-identification using Probabilistic Time-space Diagram

## Contributors:
- **Louis Sungwoo Cho, Civil & Environmental Engineering (Transportation) Major, Computer Science Minor, </br> University of Illinois at Urbana-Champaign (UIUC)**
- **Alireza Talebpour, Assistant Professor Civil & Environmental Engineering (Transportation), </br> University of Illinois at Urbana-Champaign (UIUC)**

## [UIUC CEE 497 Louis Sungwoo Cho Senior Thesis](https://lotlouischoitslab.github.io/static/media/Louis_CEE497_Thesis.93a25b1773e42226928a.pdf)

## Algorithm:
First, the trajectory points for x and y coordinates are parsed in. The points are then filtered so we do not have any empty arrays in the nested data file. For each of the predDataloader, se have the mean x coordinates, mean y coordinates, and the standard deviation of x and y coordinates. We feed those in with the x and y coordinates of each trajectory into a helper function which will calculate the line integral and return the best trajectory. The best trajectory is determined by returning the maximum line integral value. 
 
## HighwayNet Six Maneuver Architecture

## Overview
This model, `highwayNet_six_maneuver`, is designed for trajectory prediction in autonomous driving scenarios. It incorporates both the dynamics of individual vehicles and the social interactions between multiple vehicles using a convolutional social pooling mechanism.

## Core Components

### Initialization (`__init__` method)
- **Arguments Unpacking:** The method starts by unpacking various arguments from the `args` dictionary, which are configurations for the model.
- **Network Layers Definition:**
  - **Input Embedding Layer:** A linear layer to embed the input coordinates.
  - **Encoder LSTM:** An LSTM to encode the input trajectory.
  - **Vehicle Dynamics Embedding:** A linear layer to embed the output of the encoder LSTM.
  - **Convolutional Social Pooling Layers:** Convolutional layers to process the social context (neighboring vehicles).
  - **Decoder LSTM:** An LSTM to decode the concatenated embeddings (social + dynamics + maneuver).
  - **Output Layers:** Linear layers for predicting future trajectories and maneuver classes.
  - **Activations:** Various activation functions like LeakyReLU, ReLU, and Softmax.

### Forward Pass (`forward` method)
- **History Encoding:**
  - Embeds and encodes the input trajectory using the encoder LSTM.
- **Neighbor Encoding:**
  - Embeds and encodes the neighboring vehicles' trajectories.
- **Masked Scatter:**
  - Constructs a social encoding tensor by placing the neighbor encodings into a predefined grid using masks.
- **Convolutional Social Pooling:**
  - Applies convolutional layers followed by max pooling to the social encoding tensor.
- **Concatenation:**
  - Concatenates the social encoding and the history encoding.
- **Maneuver Recognition (Optional):**
  - Predicts maneuver classes if `use_maneuvers` is True.
  - During training, it concatenates the true maneuver encoding with the concatenated embedding and decodes the future trajectory.
  - During testing, it predicts trajectories for each possible maneuver class.
- **Decoding:**
  - Decodes the concatenated embedding into future trajectory predictions.

### Decoding (`decode` method)
- Repeats the concatenated embedding for the output length.
- Decodes the repeated embedding using the decoder LSTM.
- Applies the output activation function to get the final predicted trajectory.

## Model Description

**Model Architecture:**
- The input trajectories are first embedded into a higher-dimensional space using a linear layer. These embeddings are then processed by an LSTM to capture temporal dependencies.
- The social interactions are captured using a series of convolutional layers applied to the embeddings of neighboring vehicles. This is followed by max pooling to reduce the dimensionality.
- The encoded representations of the target vehicle and its neighbors are concatenated to form a comprehensive context vector.

**Maneuver Recognition:**
- The model can predict different maneuver classes (such as lane changes) and use these predictions to condition the future trajectory predictions. During training, it uses the true maneuver classes, while during testing, it considers all possible maneuvers.

**Decoding:**
- The concatenated context vector is decoded using another LSTM to generate future trajectory predictions. This process is repeated for each time step in the prediction horizon.

**Flexibility and Performance:**
- The model's architecture is flexible, with configurable parameters for different components, making it adaptable to various scenarios. The convolutional social pooling mechanism enables the model to effectively capture the interactions between multiple vehicles, improving the accuracy of trajectory predictions.

# Training the Convolutional Social Pooling Model 

## Overview
The training process involves pre-training the model using Mean Squared Error (MSE) loss and then training it using Negative Log-Likelihood (NLL) loss. The model uses both individual vehicle dynamics and social interactions to predict future trajectories.

## Core Components

### Setup and Initialization
- **Environment Configuration:** Sets the GPU to be used for training.
- **Network Arguments:** Specifies various hyperparameters and configurations for the network.
- **Network Initialization:** Initializes the CSP model with six maneuvers.
- **Optimizer:** Uses the Adam optimizer for training.
- **Data Loaders:** Loads the training and validation datasets using the TGSIM dataset class.

### Training Process
The training is divided into two phases:
1. **Pre-training with MSE Loss:** Helps the model converge faster by minimizing the mean squared error of the predicted trajectories.
2. **Training with NLL Loss:** Optimizes the model using negative log-likelihood loss for multi-modal trajectory prediction.

### Key Functions and Steps
- **Forward Pass:** Computes the predicted future trajectories and maneuver probabilities.
- **Loss Calculation:** Calculates the loss based on the current phase (MSE or NLL) and updates the model weights.
- **Validation:** Evaluates the model performance on the validation set after each epoch.






<!--- 
## HAL Cluster Notes:

                  File Name: Trajectory_Prediction.py

## NOTES For Louis Sungwoo Cho NCAS HAL Cluster:
**[Reference Video Link](https://www.youtube.com/watch?v=l1dV25xwo0o&list=PLO8UWE9gZTlCtkZbWtEcKgxYVVLIvN2IS&index=1)**


         
We are using CEE497 conda environment and go here for more **[reference](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster)**
Make sure to upload the files to the cluster if you have made any changes.

We need to: 

         conda install -c "conda-forge/label/cf202003" libopenblas
         
To connect to NCSA Hal Cluster: 

         ssh louissc2@hal.ncsa.illinois.edu
         
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
         swrun -p gpux1 -r louissc2
         
## Exit the terminal:
-->
         exit 



