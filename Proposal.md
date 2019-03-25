# 11-785 Deep Learning Project Proposal

## Please describe in a few sentences the task you propose to address
- CycleGAN

  CycleGAN has achieved a great result in unsupervised style transfer recently. However, if we want to transfer a horse to zebra, it applies the stripes to not only the horse body, but also the background and even the human riding the horse. We suppose the CycleGAN network could render better result if we provide it with an instance-level segmentation of the input image. 

- Mask R-CNN

  Mask R-CNN is widely used for object detection and instance segmentation. It can be applied to resolve the ambiguity in the image for CycleGAN to generate more precise transformation. We plan to combine the two neural networks, by feeding in the segmentation of the input image (from Mask R-CNN) into CycleGAN in addition to the input image to handle more complex cases of style transfer.

---
## Please refer some papers addressing a similar task
- GAN

  A generative model that is trained via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.

- CycleGAN

  Contrary to GAN with only one generator from input to generated image, CycleGAN adds a second generator which converts a generated image back. So there is a constraint as it needs to keep the difference between input and converted back image down. 
**Limitation**: CycleGAN fails to identify objects within output images. For example, when trying to convert an image of horse to an image of zebra, the saturation of the output background will be reduced because of the black and white stripes on zebras. Our project plans to reduce this side effect so that only the object that needs to be transferred with the style will be rendered. 

- Mask R-CNN

  They present a conceptually simple, flexible and general framework for object instance segmentation. The applied approach efficiently detects objects in image while simultaneously generating a high-quality segmentation mask for each instance.

---
## What data do you intend to use			
- ImageNet

  ImageNet is an image database organized according to the WordNet hierarchy, in which each node of the hierarchy is depicted by hundreds and thousands of images. We will use ImageNet to train CycleGAN for unsupervised style transfer.

- COCO

  COCO is a large-scale object detection, segmentation, and captioning dataset. It provides tags such as super pixel stuff segmentation and recognition in context with 80 object categories and 91 stuff categories. We are going to use COCO to train an R-CNN to identify objects in complex cases for style transfer problem.

---
## Please describe your proposed approach for the project
- Training

  1. Apply segmentation to the input image with Mask R-CNN. 
  2. Then we concatenate the input image with a new channel of the pixel-by-pixel semantic segmentation result.
  3. Finally train CycleGan with images input containing segmentation information.

- Inferencing:

  1. Apply segmentation to the input
  2. Concatenate the input with semantic segmentation and do the style-transfer

---
## Did you think about the feasibility of your task			
- We can adopt a Mask R-CNN that was pre-trained on COCO dataset.

  We will need to modify the architecture of CycleGAN as there is more channels in the input. For starters, we can train a model for horse <---> zebra conversion with a manageable portion of the imagenet. 

- And test our model on some cases where CycleGAN will make mistakes and see if Mask R-CNN would amortize those mistakes. The original cycleGAN will be used as a baseline with which we will compare our model.

