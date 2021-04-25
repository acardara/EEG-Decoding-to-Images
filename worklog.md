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
- modified ac gan to work on rgb images
- cleaned up 64x64x3 images/made them more easily accessable

