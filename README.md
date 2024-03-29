
Author: Lis Kanashiro Pereira

## ALICE

Alice is an adversarial training method built on top of the mtdnn framework: https://github.com/namisan/mt-dnn . Please check how to use it first. 

### BERT and Transformers' models

Recommended references:

1) BERT paper: https://aclanthology.org/N19-1423.pdf

2) Huggingface: https://huggingface.co/docs/transformers/model_doc/bert

3) Natural Language Processing with Transformers book: https://transformersbook.com/

## Quick start

We provide a quick example on how to run it on an English temporal task (MC-TACO). For details about the MC-TACO dataset, please check the following paper: https://aclanthology.org/D19-1332/

The examples were ran on the DGX-1 machine. 

## Environment Setup

1) First, download the ALICE package from the DGX-1 data folder:

```> /data/lis/alice/```

or directly from github:

```> git clone https://github.com/lis-kp/alice.git ```

2) Download the MC-TACO from the DGX-1 data folder: 

```>/data/lis/datasets/mctaco_dataset```
    
   and place it inside the alice/mt-dnn-alice folder.

3) Download the English RoBERTa model folder from `the DGX-1 data folder: 

    ```>/data/lis/roberta``` 
    
    and place them inside the alice/mt-dnn-alice folder.

4) Pull docker: 

    ```>docker pull allenlao/pytorch-mt-dnn:v0.5```

5) Set the environment variables:

    ```>IMAGE=allenlao/pytorch-mt-dnn:v0.5```
    
    ```>V_DIR=your_alice_directory_path```

6) Run docker:

    ```>docker run --runtime=nvidia --device=/dev/nvidia0  --device=/dev/nvidia1 --device=/dev/nvidia2 --device=/dev/nvidia3 --device=/dev/nvidia4 --device=/dev/nvidia5 --device=/dev/nvidia6 --device=/dev/nvidia7  -it  --rm --name alice_tutorial --net host --volume $V_DIR:/home/your_home_folder_name/temp  --interactive --tty $IMAGE /bin/bash```

7) Go to the project folder:
    
    ```>cd /home/your_home_folder_name/temp/mt-dnn-alice```
    
8) Example:

    I give an example with the above commands for my user on the DGX-1 machine. My username is "lis", so the commands will look like the following:
    
    ```>V_DIR=alice```
    
    ```>docker run --runtime=nvidia --device=/dev/nvidia0  --device=/dev/nvidia1 --device=/dev/nvidia2 --device=/dev/nvidia3 --device=/dev/nvidia4 --device=/dev/nvidia5 --device=/dev/nvidia6 --device=/dev/nvidia7  -it  --rm --name alice_tutorial --net host --volume $V_DIR:/home/lis/temp  --interactive --tty $IMAGE /bin/bash```

    ```>cd /home/lis/temp/mt-dnn-alice```
    
## Data Pre-processing

MC-TACO

To pre-process the dataset, run the following command:

```>python prepro_std.py --model roberta --roberta_path roberta/roberta  --root_dir mctaco_dataset/ --task_def experiments/mctaco/mctaco_task_def.yml --do_lower_case $1```


## Training with ALICE

We provide running scripts for 2 settings: standard fine-tuning, and ALICE.

### Training on MC-TACO

1) Standard fine-tuning: 

    please go to the mctaco scripts folder: 
    
    ```>cd scripts/mctaco_roberta_base```
    
    run the training script:
    
    ```>sh run_mctaco.sh```
    
2) ALICE:
     
    --------------------------------------------

    Running ALICE on RoBERTa_BASE model

    --------------------------------------------

    please go to the mctaco_alice_roberta_base scripts folder: 
    
    ```>cd scripts/mctaco_alice_roberta_base```
    
    run the training script:
    
    ```>sh run_mctaco.sh```
   
    Note that the difference is that we add the --virtual_teacher parameter in the training command.

    --------------------------------------------

    Running ALICE on RoBERTa_LARGE model

    --------------------------------------------

    please go to the mctaco_alice_roberta_large scripts folder:

    ```>cd scripts/mctaco_alice_roberta_large```

    run the training script:

    ```>sh run_mctaco.sh```

After training, we obtain the following results: 

	Standard fine-tuning (RoBERTa_BASE) ->  F1-Score:  86.04

               ALICE (RoBERTa_BASE) ->  F1-Score:  88.09

               ALICE (RoBERTa_LARGE) -> F1-Score: 90.22
    
The models are saved inside the checkpoints folder, located in the same folder as the .sh file. The models files are the files with the .pt extension, e.g. model_9.pt.

## Questions

If you have any question, please submit it on github, or send an email to kanashiro.pereira@ocha.ac.jp

## Citation
See the following paper:

Lis Pereira, Xiaodong Liu, Fei Cheng, Masayuki Asahara, Ichiro Kobayashi. Adversarial Training for Commonsense Inference. ACL2020 RepL4NLP. https://aclanthology.org/2020.repl4nlp-1.8.pdf
