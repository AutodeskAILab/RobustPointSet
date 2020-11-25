# RobustPointSet
A benchmark dataset to facilitate augmentation-independent robustness analysis of point cloud classification models. RobustPointSet comes with 6 different transformations: Noise, Translation, Missing part, Sparse, Rotation, and Occlusion.

<div align="center">
<img src="https://github.com/AutodeskAILab/RobustPointSet/blob/main/RobustPointSet.png" width="800" height="320">
</div>


### Evaluation Strategies

We test two different evaluation strategies on more than 10 models:

#### Strategy 1 (training-domain validation)
For this strategy, we train on `train_original.npy` without applying any data-augmentation, and test on each test set (i.e. `test_*.npy` ) separately.

#### Strategy 2 (leave-one-out validation)
For this strategy, each time we concatenate 6 train sets (i.e. the `train_*.npy` ones), and test on the test set (i.e. `test_*.npy` ) of the taken-out group. We repeat this process for all the groups. For example, we train with concatenation of `{train_original.npy, train_noise.npy, train_missing_part.npy, train_occlusion.npy, train_rotation.npy, train_sparse.npy}` and test on `test_translate.npy`. Similar to strategy 1, we don't apply any data-augmentation here. For both the strategies, the same label files can be used i.e. `labels_train.npy` and `labels_test.npy`.

-----------------

### Benchmarks 

| Type | Method              | Original  |   Noise   | Translation | Missing part |  Sparse   | Rotation | Occlusion |  Average  |
|:----:|:--------------------|:---------:|:---------:|:-----------:|:------------:|:---------:|:--------:|:---------:|:---------:|
|General   | PointNet            |   89.06   | **74.72** |    79.66    |    81.52     | **60.53** |   8.83   |   39.47   | **61.97** |
|General      | PointNet++ (MSG)    |   91.27   |   5.73    |    91.31    |    53.69     |   6.65    |  13.02   |   64.18   |   46.55   |
|General      | PointNet++ (SSG)    |   91.47   |   14.90   |    91.07    |    50.24     |   8.85    |  12.70   |   70.23   |   48.49   |
|General      | DGCNN               | **92.52** |   57.56   |  **91.99**  |    85.40     |   9.34    |  13.43   | **78.72** |   61.28   |
|General      | PointMask           |   88.53   |   73.14   |    78.20    |    81.48     |   58.23   |   8.02   |   39.18   |   60.97   |
|General      | DensePoint          |   90.96   |   53.28   |    90.72    |    84.49     |   15.52   |  12.76   |   67.67   |   59.40   |
|General      | PointCNN            |   87.66   |   45.55   |    82.85    |    77.60     |   4.01    |  11.50   |   59.50   |   52.67   |
|General      | PointConv           |   91.15   |   20.71   |    90.99    |    84.09     |   8.65    |  12.38   |   45.83   |   50.54   |
|General      | Relation-Shape-CNN  |   91.77   |   48.06   |    91.29    |  **85.98**   |   23.18   |  11.51   |   75.61   |   61.06   |
|RotInv      | SPHnet              |   79.18   |   7.22    |  **79.18**  |     4.22     |   1.26    |  79.18   |   34.33   |   40.65   |
|RotInv      | PRIN                |   73.66   |   30.19   |    41.21    |    44.17     |   4.17    |  68.56   |   31.56   |   41.93   |

-----------------

### Publication 

[RobustPointSet: A Dataset for Benchmarking Robustness of Point Cloud Classifiers](https://arxiv.org/abs/2011.11572)
```
@article{taghanaki2020robustpointset,
      title={RobustPointSet: A Dataset for Benchmarking Robustness of Point Cloud Classifiers}, 
      author={Saeid Asgari Taghanaki and Jieliang Luo and Ran Zhang and Ye Wang and Pradeep Kumar Jayaraman and Krishna Murthy Jatavallabhula},
      year={2020},
      journal={arXiv preprint arXiv:2011.11572}
}
```
-----------------
### Download
The dateset consists of two parts: [Part I](https://github.com/AutodeskAILab/RobustPointSet/releases/download/v1.0/RobustPointSet.z01) and [Part II](https://github.com/AutodeskAILab/RobustPointSet/releases/download/v1.0/RobustPointSet.zip). Please download both parts and unzip Part I, which will automatically extract the two parts into the same folder. 

-----------------
### License

Please refer to the [dataset license](https://github.com/AutodeskAILab/RobustPointSet/blob/main/LICENSE.md).


