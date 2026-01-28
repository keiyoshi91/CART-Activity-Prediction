# Enhancing CAR-T cell activity prediction via fine-tuning protein language models with generated CAR sequences

This repo contains code for the pape [Enhancing CAR-T cell activity prediction via fine-tuning protein language models with generated CAR sequences]().

### Abstract
 **Background**: Chimeric antigen receptor (CAR)-T cell therapy has shown remarkable success in treating hematological malignancies. However, several challenges remain, including limited efficacy against solid tumors, T cell exhaustion, and lack of T cell persistence, which have restricted its clinical efficacy across various indications. Sequence optimization of CAR constructs offers a promising strategy for enhancing the therapeutic efficacy of CAR-T cells. Recent advances in machine learning, particularly in protein language models (PLMs), have enabled the prediction of mutational effects based on sequence representations. However, applying PLMs to CARs is challenging because of the artificial nature of CARs and the absence of comprehensive CAR sequence databases. <br />
**Results**: We developed a computational framework for predicting CAR-T cell activity by fine-tuning ESM-2 with CAR sequences generated using sequence augmentation. The CAR sequences were constructed through the in silico recombination of the homologous domains of the CARs, enabling a task-specific adaptation of the model. To evaluate the prediction performance, we experimentally assessed the cytotoxicity of CAR-T cells expressing mutated CAR variants and compared the results with model predictions. Our results demonstrated that fine-tuning ESM-2 significantly improved the prediction performance of CAR-T cell activity. Furthermore, we showed that training parameters such as sequence diversity, number of training steps, and model size substantially influenced prediction performance.<br />
**Conclusions**: Our findings highlight the potential of combining sequence augmentation with fine-tuning of PLMs to advance data-driven CAR-T cell design.


## Setup
The project is confirmed to be compatible with Python 3.10.3 and CUDA 12.8.


## Code Formatting
This repository follows a standardized code formatting guideline to ensure code consistency and readability. We use the following tools for formatting Python code:

- <u>black</u>: A robust and uncompromising code formatter for Python.
<br />
- <u>isort</u>: A tool to sort and organize Python imports automatically.

Formatting Instructions
If you make changes to the code, please ensure that you apply the correct formatting using these tools before committing your changes. Below are the commands to format the code and imports:

### 1. Format the code with black:
```
black .
```
### 2. Sort imports with isort:
```
isort .
```
<br />
Please reach out to Kei Yoshida, kei.yoshida.qp@hitachi.com for any issues, comments, questions or suggestions.
