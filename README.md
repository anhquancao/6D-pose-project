# 6D object pose estimation

## Code Structure (Code that I implemented)

* **datasets**
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: class DepthDataset, slightly modified class PoseDataset
* **lib**
	* **lib/loss.py**: class LossConf and LossNoConf
	* **lib/loss_depth.py**: All codes in this file
    * **lib/network.py**: class UpProjBlock, UpProjBlockv2, R, MFF, Decoder, DepthV3, R2, DepthV4, DepthV2, DepthNetPSP, ConfNet, PoseNetRGBOnlyV2, PoseNetRGBOnly
	* **lib/utils.py**: functions im_convert, depth_to_img, visualize
	* **lib/evaluate.py**: functions to evaluate the quantitative performance of the depth estimation model.

* **Notebook in the root folder**
	* **confidence pred.ipynb**: Train ConfNet
    * **depthv1.ipynb**: Train DepthV1 model
    * **depthv2.ipynb**: Train DepthV2 model
    * **depthv3.ipynb**: Train DepthV3 model
    * **Eval_rgb_only confidence.ipynb**: Evaluate PoseNetRGBOnly with modified Confidence estimation: Bandit, ConfNet
    * **Eval_rgb_only.ipynb**: Evaluate PoseNetRGBOnly model.
    * **Eval depth.ipynb**: Evaluate the three depth estimation models.
    * **Predict depth.ipynb**: Predict the depth from RGB images and store results
    * **Training_rgb_only.ipynb**: Train PoseNetRGBOnly model
    * **Training.ipynb**: Train Original model from the paper.

## Code Structure (From Paper)

* **datasets**
	* **datasets/ycb**
		* **datasets/ycb/dataset.py**: Data loader for YCB_Video dataset.
		* **datasets/ycb/dataset_config**
			* **datasets/ycb/dataset_config/classes.txt**: Object list of YCB_Video dataset.
			* **datasets/ycb/dataset_config/train_data_list.txt**: Training set of YCB_Video dataset.
			* **datasets/ycb/dataset_config/test_data_list.txt**: Testing set of YCB_Video dataset.
	* **datasets/linemod**
		* **datasets/linemod/dataset.py**: Data loader for LineMOD dataset.
		* **datasets/linemod/dataset_config**: 
			* **datasets/linemod/dataset_config/models_info.yml**: Object model info of LineMOD dataset.
* **replace_ycb_toolbox**: Replacement codes for the evaluation with [YCB_Video_toolbox](https://github.com/yuxng/YCB_Video_toolbox).
* **trained_models**
	* **trained_models/ycb**: Checkpoints of YCB_Video dataset.
	* **trained_models/linemod**: Checkpoints of LineMOD dataset.
* **lib**
	* **lib/loss.py**: Loss calculation for DenseFusion model.
	* **lib/loss_refiner.py**: Loss calculation for iterative refinement model.
	* **lib/transformations.py**: [Transformation Function Library](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
    * **lib/network.py**: Network architecture.
    * **lib/extractors.py**: Encoder network architecture adapted from [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch).
    * **lib/pspnet.py**: Decoder network architecture.
    * **lib/utils.py**: Logger code.
    * **lib/knn/**: CUDA K-nearest neighbours library adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda).
* **tools**
	* **tools/_init_paths.py**: Add local path.
	* **tools/eval_ycb.py**: Evaluation code for YCB_Video dataset.
	* **tools/eval_linemod.py**: Evaluation code for LineMOD dataset.
	* **tools/train.py**: Training code for YCB_Video dataset and LineMOD dataset.
* **experiments**
	* **experiments/eval_result**
		* **experiments/eval_result/ycb**
			* **experiments/eval_result/ycb/Densefusion_wo_refine_result**: Evaluation result on YCB_Video dataset without refinement.
			* **experiments/eval_result/ycb/Densefusion_iterative_result**: Evaluation result on YCB_Video dataset with iterative refinement.
		* **experiments/eval_result/linemod**: Evaluation results on LineMOD dataset with iterative refinement.
	* **experiments/logs/**: Training log files.
	* **experiments/scripts**
		* **experiments/scripts/train_ycb.sh**: Training script on the YCB_Video dataset.
		* **experiments/scripts/train_linemod.sh**: Training script on the LineMOD dataset.
		* **experiments/scripts/eval_ycb.sh**: Evaluation script on the YCB_Video dataset.
		* **experiments/scripts/eval_linemod.sh**: Evaluation script on the LineMOD dataset.
* **download.sh**: Script for downloading YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints.



## Datasets

This work is tested on two 6D object pose estimation datasets:

* [LineMOD](http://campar.in.tum.de/Main/StefanHinterstoisser): Download the [preprocessed LineMOD dataset](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (including the testing results outputted by the trained vanilla SegNet used for evaluation).

Download YCB_Video Dataset, preprocessed LineMOD dataset and the trained checkpoints (You can modify this script according to your needs.):
```	
./download.sh
```
