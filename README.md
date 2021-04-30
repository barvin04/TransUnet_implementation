# Medical Image Segmentation

__Team__: Autobots

__Team Members__:<br>
Abhinaba Bala _(2020701001)_<br>
Neelabh Kumar _(2020701003)_ <br>
Ruchi Chauhan _(2018711001)_ <br>
Rupak Lazarus _(2020701020)_ <br>

__Assigned TA__: Sai Soorya Rao Veeravalli<br><br>

## MidEval Presentation [File](CVmidEval.pdf)
## Final Presentation [File](CVFinalEval.pdf)

This project is undertaken as a part of the Computer Vision coursework at IIIT Hyderabad in Spring semester 2021. The main inspiration of our work is from the very recent paper of [TransUNet](https://arxiv.org/pdf/2102.04306.pdf) which merits both Transformers and U-Net, as a strong alternative for medical image segmentation.
The overview and proposal can be found [here](https://github.com/Computer-Vision-IIITH-2021/project-autobots/blob/main/ProjectProposal.pdf).

Code structure:
```
-lists
-networks
CVmidEval.pdf
ProjectProposal.pdf
README.md
ScatterringCoeff_generator.m
US_generateMasks.py
Unet.ipynb
eval.py
rc_bcet.py
train.py
utils.py
```

## Dataset
Download the dataset,   
For lung images: [here](https://github.com/v7labs/COVID-19-xray-dataset)  
For ultrasound images: [here](https://hc18.grand-challenge.org) 

## Pre-processing
To generate the scattering-coefficients for augmenting in TranUnet, use ScatteringCoeff_generator.m   
For processing the ultrasound images use the US_generateMasks.py file.  
And, for processing the lung images use rc_bcet.py file.

## TransUNet Eval
python3 eval.py --dataset Ultrasound  --batch_size 12 --model_path /ssd_scratch/cvit/rupraze/models/ultrasound/epoch_120.pth
## TransUNet + Scattering coefficients Eval
python3 eval_scatcoeff.py.py --dataset Ultrasound  --batch_size 12 --model_path /ssd_scratch/cvit/rupraze/models/scat_ultrasound/epoch_149.pth

##  Inference
To get the demo of the inference process, run the 'InferenceDemo.ipynb' file
