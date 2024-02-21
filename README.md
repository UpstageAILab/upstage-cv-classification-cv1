[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)

# 분류 셰프: 문서 속 재료로 요리하는 재미!

## Team

<table>
<tr>
<td>  <div  align=center> 👑 </div>  </td>
<td>  <div  align=center> 1 </div>  </td>
<td>  <div  align=center> 2 </div>  </td>
<td>  <div  align=center> 3 </div>  </td>
<td>  <div  align=center> 4 </div>  </td>
</tr>
<tr>
<td>  <div  align=center>  <b>신동혁</b>  </div>  </td>
<td>  <div  align=center>  <b>김도연</b>  </div>  </td>
<td>  <div  align=center>  <b>김다운</b>  </div>  </td>
<td>  <div  align=center>  <b>서상혁</b>  </div>  </td>
<td>  <div  align=center>  <b>가상민</b>  </div>  </td>
</tr>
<tr>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/c4cb11ba-e02f-4776-97c8-9585ae4b9f1d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/3d913931-5797-4689-aea2-3ef12bc47ef0"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/0f945311-9828-4e50-a60c-fc4db3fa3b9d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/a4dbcdb5-1d28-4b91-8555-1168abffc1d0"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/6c21c014-1e77-4ac1-89ac-72b7615c8bf5"  width="250"  height="300"/>  </td>
</tr>
<tr>
<td>  <div  align=center>  <a  href="https://github.com/HyeokHam">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/d-yeon">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/Daw-ny">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/devhyuk96">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/3minka">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
</tr>
</table>

  

## 0. Overview

### Environment

-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements

-   albumentations==1.3.1
-   numpy==1.26.0
-   timm==0.9.12
-   torch==2.1.0
-   torchvision=0.16.0
-   scikit-learn=1.3.2

## 1. Competiton Info

### Overview

 이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

 이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/f35917ed-effd-4c5d-8f79-10fe1718bcc7)
  
  그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/e69229b9-b3c1-443b-a5c2-2ce499667c89)

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.
-   **input** : 3140개의 이미지
-   **output** : 주어진 이미지의 클래스

### 평가 지표

F1 score는 Precision과 Recall의 조화 평균을 의미합니다. 클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가할 수 있습니다. 수식은 다음과 같습니다
 
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/253cd5a2-0806-4822-8135-e5b35b8a88e3)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/4b52b801-89df-4e6c-b86c-48219fde4c1e)
![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/6dd9eedb-2c05-46cd-a6fd-80cf19d40b42)

- [참고자료](https://www.linkedin.com/pulse/understanding-confusion-matrix-tanvi-mittal/)
Macro F1 score는 multi classification을 위한 평가 지표로 클래스 별로 계산된 F1 score를 단순 평균한 지표입니다.

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/6c6c82fb-eb02-46f8-bd31-337376d2562a)


## 2. Components

### Directory

  

## 3. Data descrption


### Dataset overview

![패스트캠퍼스  Document Type Classification pptx (1)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/38055465-ecc0-46f2-91c4-875c38028357)
![패스트캠퍼스  Document Type Classification pptx (2)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/e5ec813f-44b8-4369-87fa-4dff5e573914)
![패스트캠퍼스  Document Type Classification pptx (3)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/6c99af33-8abf-4496-8a87-ff0a940e5e1c)
![패스트캠퍼스  Document Type Classification pptx (4)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/ecb7494d-f846-4571-b706-cca340a41071)
![패스트캠퍼스  Document Type Classification pptx (5)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/33e31e7c-5534-4af8-a003-a0aa5aeb6d81)


### EDA

![패스트캠퍼스  Document Type Classification pptx (6)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/d9c24a45-2980-4200-8416-ef9e668d9ec5)
![패스트캠퍼스  Document Type Classification pptx (7)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/476d5416-eadc-48dd-ac43-95a12b00a262)


### Data Processing

![패스트캠퍼스  Document Type Classification pptx (8)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/c822d13d-cf25-4637-a90e-ef9b7634972b)
![패스트캠퍼스  Document Type Classification pptx (9)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/a847c280-b337-4742-ae96-3691756688cf)
![패스트캠퍼스  Document Type Classification pptx (10)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/678f5f1a-3646-4426-8338-27bfb2059e13)
![패스트캠퍼스  Document Type Classification pptx (11)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/a83cbe6d-a107-442b-9b3d-75954dd9b83f)


## 4. Modeling

![패스트캠퍼스  Document Type Classification pptx (12)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/6628b9de-b671-4517-9cbf-25b3684d4315)

### Model descrition

![패스트캠퍼스  Document Type Classification pptx (13)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/8c642223-94cd-4469-8b51-1e788050883a)


### Modeling Process

![패스트캠퍼스  Document Type Classification pptx (16)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/eb9f216d-e810-425b-a899-0fe49745b72b)
![패스트캠퍼스  Document Type Classification pptx (14)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/0e7da75f-3473-462e-89e6-c189066db556)
![패스트캠퍼스  Document Type Classification pptx (15)](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/147508048/cad94c3f-d34b-4009-bb7e-e21af15a325f)



## 5. Result

### Leader Board

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv2/assets/76687996/a8859348-aba1-4336-84c0-cbe5040e2712)

### Presentation
- [Google Project](https://docs.google.com/presentation/d/1RwgKMpzbraxjYqTCn4eo3yh1iWQ6-Cah/edit?usp=sharing&ouid=107968498421720497028&rtpof=true&sd=true)

  

## etc

### Meeting Log

- 전체적인 내용은 [Notion](https://quickest-asterisk-75d.notion.site/1-e1916b7fb9b94e948381794c3b824036), [Notion2](https://www.notion.so/Document-Type-Classification-b01886bae17c4dd9b2d3244429f56fee?pvs=4)에서 확인하실 수 있습니다.
- Feb 5 ~ Feb 20 : Online Meeting

### Reference
**_PyImageSearch - image classification_**  
-   **_Pytorch Image classification_**
	1.  [https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

-   **_Image alignment_**
	2.  [https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/](https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/)

-   **_How to build a DL image dataset_**
	3.  [https://pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/](https://pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)

-   **_Image Classification and Transfer Learning_**
	4.  [https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

-   **_CNN Training_**
	5.  [https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)

- [Albumentation](https://lcyking.tistory.com/80)
