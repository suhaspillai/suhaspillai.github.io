---
layout: post
title: Creating SEM Digital twin using Deep Leaning 
---

The most highly valued companies in the world, whether Amazon, Apple, Facebook, Google, Walmart, or Netflix, have one thing in common: data is their most valuable asset. All of these companies have put that data to work using deep learning (DL). However, acquiring large quantities of data is not possible in certain industries especially in semiconductor industry. Often a company working on DL requires to generate input data that comes from customer or gather existing customer data that is usually proprietary and cannot  be released. Replicating  everything  that the customer has to generate the data is also expensive,  time-consuming,  and  the company does not have the right expertise to operate the tools in any case. Having a digital twin of the customer’s process enables a fast and inexpensive way to generate that data.

A digital twin is a digital replica of an actual physical process, system, or device. A simulation based digital twin that replicates the entire customer's process can be prohibitive. DL provides an answer with DL-based digital twin.

Here we create DL-based digital twin for mask SEM images. These mask SEM images are taken to analyze defects by expert application engineers after masks are printed. Following is an examples of [CAD](https://en.wikipedia.org/wiki/Electronic_design_automation) and its corresponding mask [SEM](https://en.wikipedia.org/wiki/Scanning_electron_microscope) image.

<figure>
<p align="center">
<div class = "column">	
  <img src="{{ site.baseurl }}/images/img1_cad.png">
</div>
<div class="column">  
  <img src="{{ site.baseurl }}/images/img1_sem.png">
</div>
</p>
<figcaption>
<b>	
<p style="color:black;font-size:15px" align="center"> Left: Mask CAD image & Right: SEM image </p>
</b>
</figcaption>
</figure>

The problem we are trying to solve is generating a SEM image given CAD image. At the first glance the problem looks similar to neural style transfer [NST](https://en.wikipedia.org/wiki/Neural_Style_Transfer) but after many iterations of modifying the architecture, loss function and training procedure we were not able to generate SEM images that resemble like real SEM images. Thus, we use ideas from [pix2pix](https://arxiv.org/abs/1611.07004) architecture. pix2pix is a conditional generative adversarial network that showed promising results in many different tasks like Labels to Street Scene, Aerial images to google maps or BW photos to colour photos etc.

We modify pix2pix architecture to generate SEM image given CAD image. The architecture involves discrimnator and generator. The goal of generator is to fool the discriminator by generating fake SEM images given CAD images. The goal of discriminator is to classify correctly between real SEM and SEM images generated by generator. The discriminator minimizes its loss by classifying images generated by generator as not real. On the other hand, the generator tries to minimize its loss by generating images that look like real SEM images, which fools the discriminator. As the training proceeds and stabilizes (which is difficult when it comes to training GANs) we can see the generator generates images that look similar to real SEM images and discriminator has a hard time to distinguish between real SEM and SEM image generated by generator.

<figure>
<p align="center">
<img src="{{ site.baseurl }}/images/sem_digital_twin_training.png">
</p>
<figcaption>
<b>
<p style="color:black;font-size:15px" align="center"> Diagram showing training SEM digital twin [1] </p>
</b>
</figcaption>
</figure>


Below figure shows image generated by the SEM digital twin and the real SEM image. The image intensity on a horizontal cutline at the same location are shown as well. Note that not only do the images look very similar, but also the signal response on edges are similar as well

<figure>
<p align="center">
<img src="{{ site.baseurl }}/images/real_img_and_DT_image.png">
</p>
<figcaption>
<b>
<p style="color:black;font-size:15px" align="center"> Left: Digital twin SEM image and signal response plot & Right: Real SEM image and signal response plot [1]</p>
</b>
</figcaption>
</figure>




Thus, creating mask SEM digital twin can help us in creating millions of SEM images. These SEM images can be used by other DL applications like DL based defect classification using SEM images. Like [Imagenet](http://image-net.org/) - we can now create SEM based imagenet, where we can create different SEM images with varying contrast, brightness, noise & edge roughness. Myriad of possibilities open up, once we have SEM based digital twin, we will see how its used in SEM defect classification in another blog.

<figure>
<p align="center">
<img src="{{ site.baseurl }}/images/sem_digital_twin_imagenet.png">
</p>
<figcaption>
<b>
<p style="color:black;font-size:15px" align="center"> Like ImageNet we can create SEM ImageNet with millions of SEM images [2]</p>
</b>
</figcaption>
</figure>



References:

[[1] Making-Digital-Twins-using-the-Deep-Learning-Kit](https://design2silicon.com/wp-content/uploads/2019/11/2019-BACUS-Making-Digital-Twins-using-the-Deep-Learning-Kit-Final.pdf)

[[2] A-deep-learning-mask-analysis-toolset-using-mask-SEM-digital-twins](https://cdle.ai/wp-content/uploads/2020/10/A-deep-learning-mask-analysis-toolset-using-mask-SEM-digital-twins.pdf)

[[3] Digital Twins: Bridging the Data Gap for Deep Learning Success](https://www.eetimes.com/digital-twins-bridging-the-data-gap-for-deep-learning-success/#)





