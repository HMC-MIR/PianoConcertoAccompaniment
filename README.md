# Piano Concerto Accompaniment

This repository contains the code for the ICASSP 2025 paper "Dense-Sparse Dynamic Time Warping for Customizing Piano Concerto Accompaniments."

The goal of this project is to generate an orchestral accompaniment recording that is time-scale modified to match a user's playing without requiring a symbolic representation of the musical piece.  

One contribution of this paper is to introduce a framework, dataset, and benchmark to evaluate the concerto accompaniment generation task.  This framework can be used to study both the offline and online accompaniment generation tasks.  In this study, we focus on piano concertos and the offline formulation of the task.  

Our main technical contribution is a novel alignment algorithm called Dense-Sparse DTW that is robust to additive noise.  It identifies prominent features in one sequence based on flux magnitude, and then aligns the selected features against the other sequence.  We show that it performs comparably to much more complicated formulations involving source separation models.

You can find the ISMIR paper [here](https://drive.google.com/file/d/1EFPm45EYrwwHNQMj8y1gY0FSZDjQUNJl/view) and the code implementing Dense-Sparse DTW [here](/System_PairwiseSparseDTW.ipynb).

## Citation

TJ Tsai, Kavi Dey, Yigitcan Ozer, and Meinard Mueller. "Dense-Sparse Dynamic Time Warping for Customizing Piano Concerto Accompaniments" in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025, to appear.

