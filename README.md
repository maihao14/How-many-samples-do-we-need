# How Many Samples Do We Need? 
## Investigating the Impact of Dataset Architecture on Deep Learning Models' Performance for Automated Phase-Picking

This repository contains the Python scripts, deep learning models, and the new training dataset for the Mackenzie Mountains used in the research paper titled _"Investigating the Impact of Dataset Architecture on Deep Learning Models' Performance for Automated Phase-Picking"_.

### Abstract

The use of deep learning (DL) in earthquake detection and phase-picking tasks has produced transformative results in recent years. Driven by large seismic datasets, DL pickers hold much promise in improving the accuracy of automated picks. However, the regionalization of seismic velocity and attenuation models makes the application of pre-established phase pickers to new target regions challenging if the input seismic data distribution is not reflected in the original training datasets. Furthermore, transfer learning of DL pickers may not be possible due to the lack of reliable human-reviewed waveforms for training. Perhaps the greatest challenge is that seismologists have no a priori knowledge of the number of waveforms required for model training to achieve their desired phase-picking accuracy and model residuals, or which proposed DL pickers can be applied directly to a new target region without re-training. In this study, we explore the issues of DL model performance by investigating the effect of increasing training sample sizes and examining different deployment settings applied to new data. To this end, we retrain two of the most popular DL pickers, PhaseNet and EQTransformer, using training datasets of various size and then test the phase picking accuracy with the same validation set. From this study, we gain insight into how many waveforms should be included in a new DL project and which additional factors (e.g., data preprocessing and standardization, picking method, tectonic setting, etc.) might affect training and model performance. Our study provides a guide for determining the optimal size of the training data set and model selection for future studies.

### Authors

- Hao Mai, Department of Earth and Environmental Sciences, University of Ottawa, Ottawa, Canada - [Email](mailto:hmai090@uottawa.ca)
- Pascal Audet, Department of Earth and Environmental Sciences, University of Ottawa, Ottawa, Canada
- H.K. Claire Perry, Canadian Hazards Information Service, Natural Resources Canada, Ottawa, Canada
- Clément Estève, Department of Meteorology and Geophysics, University of Vienna, Vienna, Austria

### Repository Contents

- **Python Scripts**: Contain the code for training and evaluating the deep learning models.
- **Deep Learning Models**: The pre-trained models used in the research (saved in `/pred` folder in this repo).
- **Training Dataset for the Mackenzie Mountains**: A dataset used for training and evaluating the models.
  
  Mackenzie Mountain Dataset can be downloaded at [here](https://www.kaggle.com/datasets/okaygoodnight/mackenzie-mountain-earthquake-dataset-for-ai-use)

### How to Use

(Provide instructions on how to use the repository, for example, how to set up the environment, how to run the scripts, etc.)

### License

MIT License

Copyright (c) 2023 Hao Mai, Pascal Audet, H.K. Claire Perry, Clément Estève

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Acknowledgements

Some of the code in this repository is revised from the [pick-benchmark](https://github.com/seisbench/pick-benchmark) repository. If you are using code from this repository for your work, please reference the following publication: [Which Picker Fits My Data? A Quantitative Evaluation of Deep Learning Based Seismic Pickers](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JB023499)
### Citation

If you find this work useful in your research, please consider citing our paper:

Mai, H., Audet, P., Perry, H.K.C., Mousavi, S.M., & Zhang, Q. (2023). Blockly earthquake transformer: A deep learning platform for custom phase picking. Artificial Intelligence in Geosciences. https://doi.org/10.1016/j.aiig.2023.05.003

```bibtex
@article{mai2023blockly,
  title={Blockly earthquake transformer: A deep learning platform for custom phase picking},
  author={Mai, Hao and Audet, Pascal and Perry, H.K. Claire and Mousavi, S. Mostafa and Zhang, Q},
  journal={Artificial Intelligence in Geosciences},
  year={2023},
  doi={10.1016/j.aiig.2023.05.003}
