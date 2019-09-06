# [Synthesizing Coupled 3D Face Modalities by Trunk-Branch Generative Adversarial Networks](https://barisgecer.github.io/files/gecer_tbgan_arxiv.pdf)
[ArXiv](https://arxiv.org/pdf/1909.02215.pdf), [Full Resolution Pdf](https://barisgecer.github.io/files/gecer_tbgan_arxiv.pdf), [Supplementary Video](https://www.youtube.com/watch?v=0M62XBn9yvM)

 [Baris Gecer](http://barisgecer.github.io)<sup> 1,2</sup>, [Alexander Lattas](https://alexanderlattas.com/)<sup> 1,2</sup>, [Stylianos Ploumpis](https://ibug.doc.ic.ac.uk/people/sploumpis)<sup> 1,2</sup>, [Jiankang Deng](https://jiankangdeng.github.io/)<sup> 1,2</sup>, [Athanasios Papaioannou](https://ibug.doc.ic.ac.uk/people/apapaioannou)<sup> 1,2</sup>, [Stylianos Moschoglou](https://ibug.doc.ic.ac.uk/people/smoschoglou)<sup> 1,2</sup>, & [Stefanos Zafeiriou](https://wp.doc.ic.ac.uk/szafeiri/)<sup> 1,2</sup>
 <br/>
 <sup>1 </sup>Imperial College London
 <br/>
 <sup>2 </sup>FaceSoft.io

<br/>
(This documentation is still under construction, please refer to our paper for more details)
<br/>

## Abstract

<p align="center"><img width="100%" src="representative.png" /></p>


Generating realistic 3D faces is of high importance for computer graphics and computer vision applications. Generally, research on 3D face generation revolves around linear statistical models of the facial surface. Nevertheless, these models cannot represent faithfully either the facial texture or the normals of the face, which are very crucial for photo-realistic face synthesis. Recently, it was demonstrated that Generative Adversarial Networks (GANs) can be used for generating high-quality textures of faces. Nevertheless, the generation process either omits the geometry and normals, or independent processes are used to produce 3D shape information. In this paper, we present the first methodology that generates high-quality texture, shape, and normals jointly, which can be used for photo-realistic synthesis. To do so, we propose a novel GAN that can generate data from different modalities while exploiting their correlations. Furthermore, we demonstrate how we can condition the generation on the expression and create faces with various facial expressions.

<br/>


## Supplementary Video

[<p align="center"><img width="100%" alt="Watch the video" title="Click to Watch on YouTube" src="https://img.youtube.com/vi/0M62XBn9yvM/maxresdefault.jpg" /></p>](https://www.youtube.com/watch?v=0M62XBn9yvM)



## Citation
If you find this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1909.02215):

```
@ARTICLE{Gecer_2019_TBGAN,
author = {{Gecer}, Baris and {Lattas}, Alexander and {Ploumpis}, Stylianos and
         {Deng}, Jiankang and {Papaioannou}, Athanasios and
         {Moschoglou}, Stylianos and {Zafeiriou}, Stefanos},
title = "{Synthesizing Coupled 3D Face Modalities by Trunk-Branch Generative Adversarial Networks}",
journal = {arXiv e-prints},
year = "2019",
month = "Sep",
eid = {arXiv:1909.02215},
pages = {arXiv:1909.02215},
archivePrefix = {arXiv},
eprint = {1909.02215},
}

```
