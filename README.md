# asdfasdf

[asdfasdf](https://soynet.io/en/) is an inference optimizing solution for AI models.

#### Features of SoyNet


## Folder Structure


```
   ├─data               : Example sample data
   ├─include            : File for using dll in Python
   ├─lib                : .dll or .so files for SoyNet
   ├─models             : SoyNet execution env
   │  └─model           : Model name
   │      ├─configs     : Model definitions (*.cfg)
   │      ├─engines     : SoyNet engine files
   │      ├─logs        : SoyNet log files
   │      └─weights     : Weight files for SoyNet models (*.weights)
   ├─samples            : Executable File
   └─utils              : Commonly-used functionand trial license
```
 - `engines` : it's made at the first time execution or when you modify the configs file.
 - `weights` : You can download .weights file from `download_soynet_weight.sh` in [weights folder](#folder-structure).
 - `license file` : Please contact [SoyNet](https://soynet.io/en/) if the time has passed.

## SoyNet Function.
 - `qweqdfsg` : Created a SoyNet handle.

   ※ `asfdasdfa`
      - `asdfasdfasdfasdf` contains parameters necessary to define the model, such as input size, engine_serialize, batch_size ...
      - The parameters required may vary depending on the model.

## Prerequisites
#### NVIDA Development Environment
 - CUDA (= 11.1)
 - cuDNN (>= 8.x)
 - TensorRT (= 8.2.1.8)
 
    ※ You need to use .dll and .so files that match CDUA and TensorRT versions. If you want another version, Please contact [SoyNet](https://soynet.io/en/).

#### S/W
 - OS : Ubuntu 18.04 LTS
 - Others : OpenCV (for reading video files and outputting the screen)

## Getting Started
Before proceeding, please refer to the [Folder Structure](#folder-structure) to see if the required files exist.


The models folder contains a detailed description.

## TABLE TEST
#### Classification
<table>
  <tr>
    <td width="200">TEST✖️✔</td>
  </tr>
</table>

<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_pytorch">EfficientNet - Pytorch</a>.</td>
    <td align=center><a href="https://github.com/lukemelas/EfficientNet-PyTorch">LINK</a>.</td>
   <td align=center>✔</td>
   <td align=center>✔</td>
   <td align=center></td>
  </tr>
    <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_TensorFlow">EfficientNet - TensorFlow</a>.</td>
    <td align=center><a href="https://github.com/qubvel/efficientnet#models">LINK</a>.</td>
   <td align=center>✔</td>
   <td align=center>✔</td>
   <td align=center></td>
  </tr>
  </tr>
    <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Inception_resnet_v2">Inception ResNet V2</a>.</td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/VGG">VGG</a>.</td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
   <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SENet_legacy_senet">SENet</a>.</td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Mobilenet_V2">MobileNet V2</a>.</td>
    <td align=center><a href="https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html">LINK</a>.</td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### Object Detection


## Model List
#### Classification
|Model|
|---|
|[EfficientNet - Pytorch](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_pytorch)|
|[EfficientNet - TensorFlow](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_TensorFlow)|
|[Inception ResNet V2](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Inception_resnet_v2)|
|[VGG](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/VGG)|
|[SENet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SENet_legacy_senet)|
|[MobileNet V2](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Mobilenet_V2)|

#### Object Detection
|Model|
|---|
|[Faster RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Faster-RCNN)|
|[RetinaFace](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/RetinaFace)|
|[EfficientDet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientDet)|
|[SSD MobileNet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SSD_Mobilenet)|
|[Yolo V3](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov3)|
|[Yolo V4](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov4)|
|[Yolo V5-l](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-l)|
|[Yolo V5-m](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-m)|
|[Yolo V5-n](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-n)|
|[Yolo V5-s](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-s)|
|[Yolo V5-x](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-x)|
|[Yolo V5-l6](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-l6)|
|[Yolo V5-m6](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.0-m6)|
|[Yolo V5 Face](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5_Face)|
|[Yolor](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolor)|

#### Object Tracking
|Model|
|---|
|[FairMot](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FairMot)|

#### Pose Estimation
|Model|
|---|
|[Pose RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Pose-RCNN)|
|[OpenPose](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Openpose-Darknet)|

#### Segmentation
|Model|
|---|
|[Mask RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Mask-RCNN)|
|[Yolact](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact)|
|[Yolact++](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact%2B%2B)|

#### GAN
|Model|
|---|
|[FAnoGan](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FAnoGan)|
|[CycleGan](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/CycleGan)|
|[pix2pix](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/pix2pix)|
|[IDN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/IDN)|
|[Glean](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/glean)|

#### NLP
|Model|
|---|
|[Transformers MaskedLM](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Transformers_MaskedLM)|

#### ETC
|Model|
|---|
|[ArcFace](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/ArcFace)|
|[Eca NFNet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Eca_NFNet)|

The model will be added continuously, so please contact [SoyNet](https://soynet.io/en/) for the **Custom Model**.

## Contact
For business inquiries or professional support requests please visit [SoyNet](https://market.soymlops.com/#/)
