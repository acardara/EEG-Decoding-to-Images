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

4/16
- added normalization of EEG data
- added result analysis (confusion matrix)
- tested different LSTM models with different frequency data (55-95 seems the best)

Week of 4/20
- added ThoughtViz classifier for best EEG wave classification accuracy so far (~30%)
- tested AC-GAN implementations
    - generates very poor 64x64x3 images

4/27
- modified AC-GAN as per recommended changes
    - Conv2DTranspose -> bilinear upsampling

4/30
- looked into GAN improvements
- added functionality Colab notebook to automatically create a new directory on Google drive and save images and model artifacts to it

May Week 1
- GAN improvements
    - tested architectures from DCGAN, AC-GAN
    - tested hyper-parameters from ^

May:
- GAN testing
    - architectures
    - datasets
    - data augmentations, etc

6/4
- added helper functions for loading datasets/augmenting loaded images

### Alexander Cardaras
4/7
- utility function for converting .npy to edf
- started visualization functions [need to verify correctness]

4/8
- mapped EEG labels to 50k ImageNet64 images. This involved mapping {eeg label -> ImageNet id}, {ImageNet id ->  ImageNet64 id}, {ImageNet64 id -> ImageNet64 class}
- utility functions for reading and writing images

4/11
- utility functions for splitting train/test set of images.
- conditional VAE implemented
    - generates low quality 12288(flattened 64x64x3) images with an average loss of 7.4k

4/13
- compiled a comprehensive list of missing images.
- web scraped and filtered images(removed stand-in images)

4/16
- created jupyter notebook for Brain2Image lstm
- added/tested gaussian noise to training set in hopes to reduce overfitting(didn't help) 

4/23
- modified ac gan(jason) to work on rgb images
- cleaned up 64x64x3 images/made them more easily accessable

4/25
- found and fixed a bug in image_utils.py related to spliting data incorrectly.
- implemented thoughtviz's baseline acgan / trained it on 10 classes of our data
    - very poor results after 100 epochs (IS ~1.3 +- 0.15)

4/27-30
- Switched over from colab to PRP cluster/ learned prp basics
- Got regular GAN(not conditional) to create qualitatively ok images
- Built and tested DCGAN
- Implemented a handful of GAN heuristics
- Added basic data augmentation to increase dataset size

5/1-5/17
- Hyperparameter testing of DCGAN and ACGAN
- Spectral Normalization on DCGAN & ACGAN
    - Hyperparameter testing with SN
- Thorough analysis of 28x28 DCGAN 
    - Hyperparameter adjustment for training(batch size, leraning rates, dataset augmentation): 
    - 
        28x28 Grayscale(1 class), 
        
        28x28 RGB(1 class),
        
        64x64 RGB(1 class) w/wo Self attention,
        
        64x64 RGB(3 classes) w/wo Self attention,
        
        64x64 RGB(10 classes)
        
