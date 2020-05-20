##ALICE

Alice is built on top of the mtdnn framework: https://github.com/namisan/mt-dnn . Please check how to use it first. 

##Quick start

At this point, we provide an example on how to run it on the MCScript2.0 dataset: https://my.hidrive.com/share/wdnind8pp5#$/. 

We preprocessed the data following the mtdnn format for ranking tasks. The data is inside the mcscript folder. To preprocess your own dataset, please check the usage of the prepro_std.py script on the documentation of the mtdnn package. 

##To run ALICE on the MCScript

1) First, download the roberta pre-trained model (and other models and datasets) by running sh download.sh
2) run sh run_alice_ranking.sh (the option --virtual_teacher enables the use of alice algorithm)

ADV baseline: To be released soon after some polish

SMART baseline: To be released soon by Microsoft Research Team.

## Citation
See the following paper:

Lis Kanashiro Pereira, Xiaodong Liu, Fei Cheng, Masayuki Asahara, Ichiro Kobayashi. Adversarial Training for Commonsense Inference. ACL2020 RepL4NLP. Arxiv version: https://arxiv.org/abs/2005.08156
