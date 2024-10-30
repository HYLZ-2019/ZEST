# [NeurIPS 2024] Zero-Shot Event-Intensity Asymmetric Stereo via Visual Prompting from Image Domain

This is code for the paper:

```
Hanyue Lou, Jinxiu Liang, Minggui Teng, Bin Fan, Yong Xu, Boxin Shi. Zero-Shot Event-Intensity Asymmetric Stereo via Visual Prompting from Image Domain. In Adv. of Neural Information Processing Systems, 2024.
```

The algorithm in the paper can be decomposed into 4 steps. Directly run `python test_algorithm.py` to test the full CREStereo + DepthAnythingV2 variant of the algorithm. If you want to test other variants, you can substitute some steps in other models. The details of the 4 steps are:

1. Convert the left-view images into temporal differential images, and convert the right-view events into temporal integral images. This step is performed in the dataset (DSEC/MVSEC/M3ED) classes. You can run this step individually and save the results using `python make_dataset.py`.

2. Use the results in step 1 and an image-based stereo vision model to predict the disparity. Only CREStereo is integrated in this code. You can use other models (such as DynamicStereo) by inferencing them using the results in step 1. The steps to load CREStereo are:
	* Load the code with `git submodule add https://github.com/ibaiGorordo/CREStereo-Pytorch.git crestereo`.
	* Download and save the model checkpoint according to the instructions in https://github.com/ibaiGorordo/CREStereo-Pytorch.

3. Use a image-based monocular model to predict depth from the left-view images. Only DepthAnythingV2 is integrated in this code. You can use other models (such as DepthAnythingV1 or MiDaS) by inferencing them using the left-view images. 

4. Merge the results from step 2 and step 3 to get final disparity predictions. You can run this step individually on saved `.npy` depth results using `python merge_results.py`.

If you find the code helpful to your research, please cite:

```
@inproceedings{lou2024zest,
  title={Zero-Shot Event-Intensity Asymmetric Stereo via Visual Prompting from Image Domain},
  author={Lou, Hanyue and Liang, Jinxiu and Teng, Minggui and Fan, Bin and Xu, Yong and Shi, Boxin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```