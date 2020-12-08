# Aiffel


- **설명 :** 영상 처리 과정에서 진행한 프로젝트를 정리하기 위한 Repository입니다.
- **기관 :** [모두의 연구소](https://modulabs.co.kr)
- **과정 :** [AIFFEL](https://aiffel.io)
- **기간 :** 2020. 07. 23 ~ 2020. 12. 23


# 프로젝트


### EXPLORATION : 다양한 분야의 프로젝트 수행을 통해 관심 분야를 찾는 과정


|No| 제목 | 학습목표 | 데이터 | 모델 |
|---|---|----|----|---|
|1|[가위바위보](https://github.com/benestump/Aiffel/tree/master/01_rock_scissor_paper)|가위, 바위, 보 이미지를 인식하여 이미지를 분류하는 기초적인 방법에 대해 배운다.|캠을 활용하여 가위, 바위, 보 이미지를 촬영하여 활용|CNN|
|2|[손글씨, 와인, 유방암 구분](https://github.com/benestump/Aiffel/tree/master/02_Iris_classification)|scikitlearn의 데이터를 이용하고 기본적인 머신러닝 모델의 사용방법과 성능 지표에 대해 배운다.|Scikitlearn breast cancer|DecisionTreeClassifier|
|3|[뉴스기사 크롤링 및 분류](https://github.com/benestump/Aiffel/tree/master/03_news_crawling_classification)|크롤링을 위한 라이브러리의 사용법을 익히고 나이브 베이즈 분류기를 사용하여 카테고리를 분류하는 모델을 만들어 본다.|네이버 뉴스 기사 크롤링|Naive Bayes Classification|
|4|[꽃종류 분류](https://github.com/benestump/Aiffel/tree/master/04_cat_dog_classification)|TensorFlow에서 제공하는 데이터를 활용하여 전이학습에 대해서 배우고 다양한 전략을 수행하는 방법을 익힌다 |tf_flowers|VGG16, DenseNet169|
|5|[자전거 이용자수 예측](https://github.com/benestump/Aiffel/tree/master/05_bike_regression)|선형 회귀의 전체 프로세스를 직접 구현하여 프로세스를 학습한다.|[Kaggel:Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data?select=test.csv)|LinearRegression|
|6|[주식 예측](https://github.com/benestump/Aiffel/blob/master/Exploration/06_stock_prediction/E6_Project_Stock_Prediction.ipynb)|시계열 데이터의 특징과 ARIMA 모델에 대해 학습하고 이를 활용하여 주식데이터를 예측한다.|finance yahoo의 주식 데이터|ARIMA|  
|7|[Movielens 영화 추천](https://github.com/benestump/Aiffel/blob/master/Exploration/07_recommend_IU/E7_Project_Recommend_Movie.ipynb)|추천 시스템의 개념과 목적을 이해한다.|Movielens|AlternatingLeastSquares|
|8|[집값 예측](https://github.com/benestump/Aiffel/blob/master/Exploration/08_housing_price_prediction/E8_House_Price_Prediction.ipynb)|캐글 경진대회에 참여하여 앙상블 개념과 다양한 방법에 대해 배운다.|[캐글 데이터](https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr)|xgboost, random forest, lightgbm(Random search)|
|10|[카메라 스티커](https://github.com/benestump/Aiffel/blob/master/Exploration/10_sticker_camera/E10_Project_sticker_camera.ipynb)|이미지에서 dlib 라이브러리를 활용하여 얼굴 인식 카메라의 흐름을 이해한다.|
|11|[작사가 만들기](https://github.com/benestump/Aiffel/blob/master/Exploration/11_lyricist/E11_lyricist.ipynb)| Sequence to Sequence에 대한 이해와 가사 데이터를 활용하여 작사를 하는 모델을 만들어본다.|[song Lyrics](https://www.kaggle.com/paultimothymooney/poetry/data)| LSTM|
|12|[닮은 연예인 찾기](https://github.com/benestump/Aiffel/blob/master/Exploration/13_tfjs_mobile/E13_face2emoji.ipynb)|임베딩에 대해 이해하고 얼굴의 임베딩 벡터를 추출하여 닮은 꼴의 연예인은 찾는 것을 실습한다.|크롤링(연예인 사진)|FaceNet|
|13|[Face2Emoji](https://github.com/benestump/Aiffel/blob/master/Exploration/13_tfjs_mobile/E13_face2emoji.ipynb)|웹과 HTML, 자바스크립트의 기본 개념을 이해하고 이를 활용한 웹 어플리케이션을 만든다.| |MobileNetV2|
|14|[Shallow Focus](https://github.com/benestump/Aiffel/blob/master/Exploration/14_shallow_focus/E14_shallow_focus.ipynb)|최신 스마트폰의 인물모드 사진의 기능을 image segmentation을 활용해 구현해 본다.| |DeepLab v3+|
|16|[CIFAR-10이미지 생성](https://github.com/benestump/Aiffel/blob/master/Exploration/16_DCGAN_newimage/E16_DCGAN_New_Image.ipynb)|생성적 적대 신경망의 구조와 원리를 이해하고 텐서플로우로 짠 DCGAN 학습 코드를 실습힌다.|CIFAR-10|DCGAN|
|17|[OCR](https://github.com/benestump/Aiffel/blob/master/Exploration/17_OCR_python/E17_OCR_python.ipynb)|OCR에 대해 이해하고 다양한 방법을 활용하여 OCR을 수행한다| |Google OCR API, keras-ocr, Tesseract|
|19|[Super Resolution](https://github.com/benestump/Aiffel/blob/master/Exploration/19_super_resolution/E19_Super_Resolution.ipynb)|Super Resolution과 그 과정 수행에 필요한 기본 개념을 이해하고 SRCNN, SRGAN를 이해하고 활용한다.| |SRCNN, SRGAN|
|20|[Anomaly Detection](https://github.com/benestump/Aiffel/blob/master/Exploration/20_anomaly_detection/E20_Anomaly_Detection.ipynb)|시계열 데이터의 이상치 개념을 이해하고 다양한 방법으로 탐색, 처리하는 방법을 배운다.|Finance Yahoo(kospi)|K-means, DBSCAN, Auto-Encoder|
|22|[Pneumonia Discrimination](https://github.com/benestump/Aiffel/blob/master/Exploration/22_Pneumonia_discrimination/E22_Pneumonia_Discrimination.ipynb)|의료 영상을 분석하는 방법을 이해하고 딥러닝을 활용하여 폐렴의 유무를 판별하는 모델을 만든다.|[캐글 Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)|CNN|
|24|[Conditional Generative Model](https://github.com/benestump/Aiffel/blob/master/Exploration/24_Conditional_Generative_Model/E24_Conditional_Generative_Model.ipynb)|조건을 부여하여 생성 모델을 다루는 방법에 대해 이해합니다.|cityscapes|Pix2Pix|
|25|[Spectrogram Classification](https://github.com/benestump/Aiffel/blob/master/Exploration/25_wav_classification/E25_Wav_Classification.ipynb)|Audio형태의 데이터를 다루는 방법을 배우고 classification모델을 직접 제작하여 본다.|waveform형태의 데이터|skip-connection을 활용한 wave classification 모델|
|26|[KorQuAD Task](https://github.com/benestump/Aiffel/blob/master/Exploration/26_KorQuAD_Task/E26_KorQuAD_Task.ipynb)|Transformer Encoder로 이루어진 BERT모델구조를 이해한다.|KorQuAD|BERT|
|27|[용종 검출](https://github.com/benestump/Aiffel/blob/master/Exploration/27_medical_UNet/E27_Medical_UNet.ipynb)|위내시경 이미지에 용종을 찾는 Segmentation모델을 만든다.|[Giana Dataset](https://giana.grand-challenge.org/Dates/)|U-Net|
|28|[Anomaly Detection with GAN](https://github.com/benestump/Aiffel/blob/master/Exploration/28_anomaly_detection/E28_Anomaly_Detection_with_GAN.ipynb)|Anomaly data가 부족한 상황에서 GAN을 이용하여 이미지 이상감지 모델을 구축하는 논리를 파악한다.|CIFAR-10|Skip-GANomaly|
|29|[Ainize 서비스](https://github.com/benestump/mnist-mlp-app)|Ainize 서비스를 활용하여 포트폴리오를 만드는 법을 익히고 기본적인 Docker 사용법을 배운다.|||
|30|[Music Transformer](https://github.com/benestump/Aiffel/blob/master/Exploration/30_music_transformer/E30_Music_Transformer.ipynb)|MIDE파일 형식을 이해하고 Music Transformer 모델을 구현하고 테스트를 수행한다.|MAESTO|Transformer|


### Going Deeper(CV) : Computer Vision 분야의 심화 프로젝트 수행 

|No|제목|학습목표|데이터|모델|
|---|---|----|----|----|
|1|[ResNet](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/01_ResNet_Ablation_Study/G1_ResNet_Ablation_study.ipynb)|직접 ResNet을 구현하며 모델을 config에 따라 변경가능하도록 설계한다.||ResNet|
|2|[Augmentation](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/02_Augmentation/G2_Augmentation.ipynb)|Augmentation 적용을 통해 학습을 시키고 최신 기법 및 활용에 대해 배운다.|tfds:Stanford dogs|ResNet50(CutMix, MixUp) |
|3|[Object Detection](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/03_Object_Detection/G3_Object_Detection.ipynb)|Object Detection을 위한 바운딩 박스 데이터셋 전처리와 활용을 배운다.|tfds:kitti|RetinaNet|
|4|[Semantic Segmentation](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/04_Semantic_Segmentation/G4_Semantic_Segmentation.ipynb)|Semantic Segmentation을 위한 데이터셋을 다루고 모델을 만드는 방법을 배운다.|KITTI|U-Net|
|5|[Class Activation Map](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/05_Class_Activation_Map/G5_Class_Activation_Map.ipynb)|Classification model로 부터 CAM을 얻어 설명 가능한 딥러닝에 대한 기본 지식을 배운다.|tfds:cars196|ResNet50|
|7|[Video Sticker App](https://github.com/benestump/Aiffel/blob/master/Going_Deeper(CV)/07_Video_Sticker_App/G07_Video_Sticker_App.ipynb)|동영상을 다루는 법을 배우고 안정적인 동작을 위한 칼만 필터에 대한 개념을 배우고 이해한다.|||
|8|[Coarse to fine](https://github.com/benestump/Aiffel/tree/master/Going_Deeper(CV)/08_Coarse_to_fine)|라벨링 툴을 만드는 방법을 익히고 색상 값을 이용한 검출 방법을 배운다.|LFW|ResNet50|


# 느낀점

- 5개월의 과정을 통해 수많은 프로젝트를 수행하고 논문 리뷰, 토론을 통해서 많은 것을 배우고 열정적인 사람들과 함께 하면서 나 자신에 대한 부족한 점을 생각해보는 시간이 되었다. 
- 아직도 배워야 할 것이 너무나 많고 앞으로도 꾸준히 공부할 것이 늘어날 것이라는 것이 막막하기도 하지만 다른 한편으로는 늘 새로운 것을 배우고 끊임없이 목표를 만들 수 있을것이라는 기대도 된다. 
- 현실의 문제를 해결하는 것은 이론을 공부하는 것만으로는 많이 부족하고 하루라도 빨리 현업에서 수많은 문제들을 해결해보고 싶은 생각이 많이 들었다. 


# 아쉬운 점 및 개선할 점 

- 공부했던 것들을 정리하고 문서화하는데 아직은 많은 부족함이 있어서 시간이 지나고 나니 아쉬움이 남는다. 많은 공부를 하였지만 체계적으로 정리를 해두지 않아 완전히 내것이 되지 못했다는 생각이 들어 안타깝다. 
- 팀프로젝트를 수행함에 있어서 경험의 부족이 많이 드러났다. 프로젝트 전체를 보는 눈이 부족하고 시간 관리, 팀원 관리 등 체계젹인 관리 방법에 대한 아쉬움이 남는다. 
- 코로나라는 특수한 상황으로 인해 한 장소에 모여 토론하고 이야기를 나누면서 학습을 수행하는 시간이 너무나 부족했던 것이 아쉽다. 



