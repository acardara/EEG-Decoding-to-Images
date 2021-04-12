# Log file for recording work allocation

### Robert Sato
4/7
- convert data to .npy files
- base colab notebook for classifying EEG waves
- LSTM with ~16% test accuracy

4/11
- helper functions for reformatting images
- GAN implementation - modified from reference noted in notebook
    - generates low quality 64x64x3 images

### Alexander Cardaras
4/7
- utility function for converting .npy to edf
- started visualization functions(need to verify correctness)

4/8
- mapped EEG labels to 50k ImageNet64 images. This involved mapping {eeg label -> ImageNet id}, {ImageNet id ->  ImageNet64 id}, {ImageNet64 id -> ImageNet64 class}
- utility functions for reading and writing images
