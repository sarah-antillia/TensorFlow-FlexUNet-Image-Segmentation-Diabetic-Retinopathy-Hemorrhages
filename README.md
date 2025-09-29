<h2>TensorFlow-FlexUNet-Image-Segmentation-Diabetic-Retinopathy-Hemorrhages (2025/09/30)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Diabetic Retinopathy Hemorrhage Lesions</b>, based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
 and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1-DaxT-7R9SiXVBQMKl9ouBctgNJcu4ey/view?usp=sharing">Augmented-Hemorrhages-PNG-ImageMask-Dataset.zip</a>
, which was derived by us from 
<a href="http://www.it.lut.fi/project/imageret">diaretdb1_v_1_1 (http://www.it.lut.fi/project/imageret)
</a>
<!--
<br>
<br>
<br>On singleclass Hemorrhages model, please refer to our repository
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages">
Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages</a>
 -->
<br>
<br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a>,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_101_0.3_0.3_image010.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_101_0.3_0.3_image010.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_101_0.3_0.3_image010.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_104_0.3_0.3_image020.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_104_0.3_0.3_image020.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_104_0.3_0.3_image020.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_105_0.3_0.3_image021.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_105_0.3_0.3_image021.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_105_0.3_0.3_image021.png" width="320" height="320"></td>
</tr>
</table>

<hr>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here was obtained from the following 
<a href="http://www.it.lut.fi/project/imageret">diaretdb1_v_1_1 (http://www.it.lut.fi/project/imageret)
</a>
<br><br>
The database consists of 89 colour fundus images of which 84 contain at least mild non-proliferative<br>
 signs (Microaneurysms) of the diabetic retinopathy, and 5 are considered as normal which do not contain<br>
  any signs of the diabetic retinopathy according to all experts who participated in the evaluation. <br>
  Images were captured using the same 50 degree field-of-view digital fundus camera with varying imaging settings.<br>
   The data correspond to a good (not necessarily typical) practical situation, where the images are comparable, <br>
   and can be used to evaluate the general performance of diagnostic methods. <br>
   This data set is referred to as "calibration level 1 fundus images".
<br><br>

<b>Licence</b><br>
Unknown
<br>
<br>
<h3>
<a id="2">
2 Hemorrhages ImageMask Dataset
</a>
</h3>
<h4>2.1 Download Hemorrhages-PNG-ImageMask-Dataset</h4>
 If you would like to train this Hemorrhages Segmentation model by yourself,
 please download  our dataset <a href="https://drive.google.com/file/d/1-DaxT-7R9SiXVBQMKl9ouBctgNJcu4ey/view?usp=sharing">
 Augmented-Hemorrhages-PNG-ImageMask-Dataset.zip  </a> on the google drive
, expand the downloaded and put it under <b>./dataset</b> folder to be.<br>
<pre>
./dataset
└─Hemorrhages
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Hemorrhages Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Hemorrhages/Hemorrhages_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large  to use for a training set of our segmentation model.
<br>
<br>
  
To generate the pre-augmented dataset, we used an offline augmentation tool 
<a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a> and
and a dataset <a href="./generator/split_master.py">splitter.py.</a><br>
<br>
On the derivation of this dataset, please see also our experiment
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages">
Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages</a>
 <br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Hemorrhages TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Hemorrhages/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Hemorrhagesand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small base_filters=16 and large base_kernels=(9,9) for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learning_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Hemorrhages 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"

;Hemorrhages rgb color map dict for 2 classes.
;   Background:black, Hemorrhages: red
rgb_map = {(0,0,0):0,(255,0,0):1}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 22,23 24)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 46,47,48)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 48.<br><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/train_console_output_at_epoch48.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Hemorrhages/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Hemorrhages/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Hemorrhages</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Hemorrhages.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/evaluate_console_output_at_epoch48.png" width="720" height="auto">
<br><br>Image-Segmentation-Hemorrhages

<a href="./projects/TensorFlowFlexUNet/Hemorrhages/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Hemorrhages/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0064
dice_coef_multiclass,0.9974
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Hemorrhages</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for Hemorrhages.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Hemorrhages/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_101_0.3_0.3_image011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_101_0.3_0.3_image011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_101_0.3_0.3_image011.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_102_0.3_0.3_image015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_102_0.3_0.3_image015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_102_0.3_0.3_image015.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_104_0.3_0.3_image020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_104_0.3_0.3_image020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_104_0.3_0.3_image020.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/barrdistorted_105_0.3_0.3_image021.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/barrdistorted_105_0.3_0.3_image021.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/barrdistorted_105_0.3_0.3_image021.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/distorted_0.02_rsigma0.5_sigma40_image010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_image010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/distorted_0.02_rsigma0.5_sigma40_image010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/images/distorted_0.03_rsigma0.5_sigma40_image018.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test/masks/distorted_0.03_rsigma0.5_sigma40_image018.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Hemorrhages/mini_test_output/distorted_0.03_rsigma0.5_sigma40_image018.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1.The diaretdb1 diabetic retinopathy database and evaluation protocol </b><br>
T. Kauppi, V. Kalesnykiene, J.-K. Kamarainen, L. Lensu, I. Sorri, A. Raninen, et al.<br>
<a href="https://webpages.tuni.fi/vision/public_data/publications/bmvc2007_diaretdb1.pdf">
https://webpages.tuni.fi/vision/public_data/publications/bmvc2007_diaretdb1.pdf</a>

<br>
<br>
<b>2.Hard Exudates Segmentation in Diabetic Retinopathy Using DiaRetDB1</b><br>
Ma Yinghua, Yang Heng, R. Amarnath, Zeng Hui<br>
<a href="https://ieeexplore.ieee.org/document/10669034">https://ieeexplore.ieee.org/document/10669034</a>
<br>
<br>
<b>3. Detection of Early Signs of Diabetic Retinopathy Based on Textural <br>
and Morphological Information in Fundus Images</b><br>
Adrián Colomer, Jorge Igual, Valery Naranjo <br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7071097/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC7071097/
</a>
<br>
<br>
<b>4.Evaluation of fractal dimension effectiveness for damage detection in retinal background</b><br>
Adrián Colomer, Valery Naranjo, Thomas Janvier, Jose M. Mossi<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0377042718300268">
https://www.sciencedirect.com/science/article/pii/S0377042718300268</a>
<br>
<br>

<b>5.Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages </b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages
</a>

