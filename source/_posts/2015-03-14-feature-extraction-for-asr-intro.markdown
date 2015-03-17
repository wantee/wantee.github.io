---
layout: post
title: "Feature Extraction for ASR: Intro"
author: Wantee Wang
date: 2015-03-14 15:49:44 +0800
comments: true
categories: [Automatic Speech Recognition]
header-includes:
   - \usepackage{graphicx}
   - \usepackage[all]{hypcap}
---

Feature extraction is the first step for Automatic Speech Recognition(ASR), which converts the waveform speech signal to a set of feature vectors. The main goal is to make the vectors have high discrimination between phonemes.

Thus, the features should be

* perceptually meaningful, i.e., analogous to features used by human auditory system.
* invariant, i.e., robust to variations in channel, speaker and
transducer.

Three main steps for feature extraction are

1. Preprocessing
2. Feature Analysis
3. Parametric Transformation

The *preprocessing* step converts the speech signal to a more suitable waveform for the following analysis, including *DC offset removal*, *pre-emphasis* and *Hamming Windowing*.

*Feature Analysis* is most important step, which do most of the works. Generally, it is can be divided into two main categories, *Spectral Analysis* and *Temporal Analysis*. Spectral analysis gives MFCC and PLP features, while temporal analysis produces Energy and Pitch features. MFCC involves *Cepstral Analysis* and PLP is based on *Linear Predictive Coding(LPC) Analysis*.

The final step, *Parameter transformation*, converts the features obtained by above step into signal parameters through *differentiation* and *concatenation*.

Details are in following posts:

1. {% post_link 2015-03-14-feature-extraction-for-asr-preprocessing %} 
2. {% post_link 2015-03-14-feature-extraction-for-asr-mfcc %}
3. {% post_link 2015-03-14-feature-extraction-for-asr-plp %}
4. {% post_link 2015-03-14-feature-extraction-for-asr-pitch %}
5. {% post_link 2015-03-14-feature-extraction-for-asr-delta %}

