# DICDDAN
Code release for domain invariant and class distinguishable feature learning using multiple adversarial networks.

If you have any problem about our code, feel free to contact fancangning@gmail.com or describe your problem in Issues.

We use the basic code of data preprocessing, basic network consrtuction and so on in CDAN.

We would like to express our sincere gratitude to the authors of CDAN for their contributions to the transfer learning community.

The parts we developed by ourselves are as follows:

In train\_image.py: cal\_A\_distance(); calweight(); getdataloader();

In loss.py: cal\_weight\_err(); proposed();

In network.py: class AdversarialNetworkClassGroup(); class AdversarialNetworkGroup();

It's worth noting that we also made a lot of changes in train() in train\_image.py to fit our method.

The rest of the work refers to CDAN.

Its code address is https://github.com/thuml/CDAN
