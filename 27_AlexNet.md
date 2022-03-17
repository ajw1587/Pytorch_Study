# AlexNet (2012)

- ILSVRC 2012년 대회에서 2위와 큰 차이로 1위 성능을 달성 → Top 5 test error: 15.4%
- 데이터셋
    - ImageNet 이용
    - 이미지 크기를 256x256으로 고정 → 추후 224x224로 crop
- 구조

    | ![space-1.jpg](Figure/AlexNet.png) |
    |:--:|
    | <b>GPU를 두개로 나눠서 학습</b>|

  - 5개의 convolution layer와 3개의 fully-connected layer
  - Response-normalization layer는 첫번째, 두번째 convolutional layer에 있음.
  - Max-pooling layer는 Response-normalization layer와 5번째 convolution layer 뒤에 위치
  - ReLu는 모든 convolution layer와 fully-connected layer 뒤에 위치
  - Input(224x224x3)-Conv1(in_channel:3, out_channels:96, 11x11, stride:4)-MaxPool1- Norm-Conv2-MaxPool2-Norm2-Conv3-Conv4-Conv5-MaxPool3-FC1-FC2-OutputLayer
  - ReLu Nonlinearity
  - Local Response Normalization
  - Dropout

# 참고 자료

- [https://deep-learning-study.tistory.com/376](https://deep-learning-study.tistory.com/376)
