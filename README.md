# The implementation for **WAVEFORM BOUNDARY DETECTION FOR PARTIOALLY SPOOFED AUDIO**

## Citation
The original paper can be found at: https://arxiv.org/abs/2211.00226


## Introduction
The first Audio Deep Synthesis Detection challenge (ADD 2022) is the first challenge to propose the partially fake audio detection task. We propose a novel framework by introducing the question-answering (fake span discovery) strategy with the self-attention mechanism to detect partially fake audios. The proposed fake span detection module tasks the anti-spoofing model to predict the start and end positions of the fake clip within the partially fake audio and finally equips the model with the discrimination capacity between real and partially fake audios. Our submission ranked second in the partially fake audio detection track of ADD 2022.


## Usage
Please see run.sh as the main entry.

This recipe provides a tiny dataset that is part of the Training, Development, and Adaptation set, for sanity testing. If you want the complete dataset, please contact the organizer of the ADD challenge (registration@addchallenge.cn).

The noise and impulse response dataset can be download from open resources:
* the MUSAN dataset: https://www.openslr.org/17/
* the simulated RIR dataset: https://www.openslr.org/26/

## Citation
