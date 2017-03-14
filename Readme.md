Image Caption using Attension - https://www.researchgate.net/project/Image-Caption-using-Visual-Attention, final Report [here](reports/Report%20-%20Image%20Captioning%20with%20Visual%20Attention.pdf)

Reference - Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." arXiv preprint arXiv:1502.03044 2.3 (2015): 5. - https://arxiv.org/abs/1502.03044

In this project, we incorporated an attention mechanism with gradient descent based training algorithms to build a deep learning model that learns to generate image descriptions given an unseen image, inspired by the state-of-the-art work in this field. Our results showed that attention based deep learning models can learn to produce good image captions and generate visualizations to infer how a model can focus on certain parts of the image while predicting the next word. We used a Convolutional Neural Network (CNN) to generate attention based features and a Long Short-term Memory Network (LSTM) for generating image captions.

Sample Visulization of Attention: 
![A dog leaps into the air in a grassy field surrounded by trees][logo]

[logo]: https://github.com/pankajb64/image_caption_using_attention/blob/master/results/dog_viz_c.png "A dog leaps into the air in a grassy field surrounded by trees" 
 
To run the attention code, run `code/notebooks/Keras Attention Trial.ipynb`

Requirements:
* Keras
* Theano
* Numpy
* Scipy
* Scikit-Image

http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/
https://keras.io/getting-started/sequential-model-guide/#examples
https://avisingh599.github.io/deeplearning/visual-qa/
https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975#file-classifier_from_little_data_script_3-py-L109

http://t-satoshi.blogspot.com/2015/12/image-caption-generation-by-cnn-and-lstm.html

https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
