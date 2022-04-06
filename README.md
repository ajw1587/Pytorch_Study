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
|TABLE TEST|
|TABLE TEST|
