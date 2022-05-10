# PRML
Python codes implementing algorithms described in Bishop's book "Pattern Recognition and Machine Learning"

## Required Packages
- python 3
- numpy
- scipy
- jupyter (optional: to run jupyter notebooks)
- matplotlib (optional: to plot results in the notebooks)
- sklearn (optional: to fetch data)

## Notebooks

The notebooks in this repository can be viewed with nbviewer or other tools, or you can use [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/), a free computing environment on AWS (prior [registration with an email address](https://studiolab.sagemaker.aws/requestAccount) is required. Please refer to [this document](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-onboard.html) for usage).

From the table below, you can open the notebooks for each chapter in each of these environments.

|nbviewer|Amazon SageMaker Studio Lab|
|:-------|:--------------------------:|
|[ch1. Introduction](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch01_Introduction.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch01_Introduction.ipynb)|
|[ch2. Probability Distributions](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch02_Probability_Distributions.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch02_Probability_Distributions.ipynb)|
|[ch3. Linear Models for Regression](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch03_Linear_Models_for_Regression.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch03_Linear_Models_for_Regression.ipynb)|
|[ch4. Linear Models for Classification](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch04_Linear_Models_for_Classfication.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch04_Linear_Models_for_Classfication.ipynb)|
|[ch5. Neural Networks](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch05_Neural_Networks.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch05_Neural_Networks.ipynb)|
|[ch6. Kernel Methods](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch06_Kernel_Methods.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch06_Kernel_Methods.ipynb)|
|[ch7. Sparse Kernel Machines](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch07_Sparse_Kernel_Machines.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch07_Sparse_Kernel_Machines.ipynb)|
|[ch8. Graphical Models](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch08_Graphical_Models.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch08_Graphical_Models.ipynb)|
|[ch9. Mixture Models and EM](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch09_Mixture_Models_and_EM.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch09_Mixture_Models_and_EM.ipynb)|
|[ch10. Approximate Inference](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch10_Approximate_Inference.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch10_Approximate_Inference.ipynb)|
|[ch11. Sampling Methods](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch11_Sampling_Methods.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch11_Sampling_Methods.ipynb)|
|[ch12. Continuous Latent Variables](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch12_Continuous_Latent_Variables.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch12_Continuous_Latent_Variables.ipynb)|
|[ch13. Sequential Data](https://nbviewer.jupyter.org/github/ctgk/PRML/blob/main/notebooks/ch13_Sequential_Data.ipynb)|[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/ctgk/PRML/blob/main/notebooks/ch13_Sequential_Data.ipynb)|

If you use the SageMaker Studio Lab, open a terminal and execute the following commands to install the required libraries.

```bash
conda env create -f environment.yaml  # might be optional
conda activate prml
python setup.py install
```
