# VGGNet(2014)

- 3x3 convolution layer를 연속적으로 사용한 16, 19 깊이의 모델 
(이전의 모델들은 오버피팅, gradient vanishing, 연산량 문제 등으로 깊이를 증가시키기가 어려웠음)
- Abstract
    - 깊이를 증가하여 정확도를 증가
    - 3x3 filter를 여러 겹 사용하여 5x5, 7x7 filter를 분해하면 추가적인 비선형성을 부여하고 parameter의 수를 감소
    - pre-initialization을 이용
    - data augmentation(resize, crop, flip)을 적용하면 다양한 scale로 feature를 포착
    - 빠른 학습을 위해 4-GPU data parallerism을 활용
- 구조
    - 입력값: 224x224 RGB 이미지  (training set의 각 pixel에 평균 RGB값을 빼줘서 전처리)
    - 3x3 filter가 적용된 CovNet, 1x1 filter를 이용하여 비선형성을 더함
    - Stride=1, padding 적용
    - 일부 conv layer에는 max-pooling (size=2x2, stride=2) 를 사용
    - 컨벌루션 레이어 뒤에는 3개의 FC layer (4096-4096-1000(softmax))
    - 모든 layer에 ReLU를 사용
    - AlexNet에 적용되었던 LRN는 적용 X

    <img src = "Figure/VGGNet.png" width=70%>

- Stride가 1인 3x3 필터만을 사용
    - 3x3 필터를 여러 개 사용하면 하나의 relu 대신 여러 개의 relu를 사용 가능
    - 5x5, 7x7을 쓸 때 보다 파라미터 수도 줄일 수 있음

# 참고 자료

- [https://deep-learning-study.tistory.com/398](https://deep-learning-study.tistory.com/398)
- [https://89douner.tistory.com/61](https://89douner.tistory.com/61)
