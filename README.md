# It's Corn (PogChamps \#3) Kaggle Competition - 1st Place Winning Solution

In this GitHub repository I am sharing my winning solution for the [It's Corn Kaggle Competition](https://www.kaggle.com/competitions/kaggle-pog-series-s01e03) along with the experiments done along the way.

## Winning Solution Details

For those who would like to jump to the relevant code, the models that were used in the winning ensemble were trained in [Experiment 6](6_-_large_model_ensemble_with_tested_augmentations.ipynb) and [Experiment 10](10_-_early_stopping_oversampling_ensemble.ipynb) (all models Test Time Augmentation (TTA) results were simply averaged).  Both experiments train ensembles of the same architecture types with different training/sampling strategies (explained below).

Experiments were conducted with the [fast.ai](https://www.fast.ai/) deep learning framework.  Starting models that were chosen were pre-trained on ImageNet.  Learning rates were obtained using the learning rate finder in the FastAI library.  Models were trained in mixed precision (fp16).  Batch sizes were effectively 64 (with gradient accumulation).  Each model was trained with a simple 80/20 train/validation split (for the most part, validation accuracy was within 1-2% of public leaderboard scores).  Image sizes were 256x256 with bordered padding (made most images look like an extended white background with not initial distortions added before batch augmentations).

Both experiments are similar in the sense that they both fine-tuned the following four model architectures from the [Py**T**orch **IM**age **M**odels (TIMM) repository](https://github.com/rwightman/pytorch-image-models):

- 'convnext_large_in22k'
- 'vit_large_patch16_224'
- 'swinv2_large_window12_192_22k'
- 'swin_large_patch4_window7_224'

with the same set of batch augmentations that were tested to do well on smaller models with this data (see [Experiment 5](5_-_identify_performance_tweaks.ipynb) if interested):

- random flip + rotate only (safe and produced the best results on all small model tests)
- random flip + rotate with small cropping allowed (added a little variation for helping the move to larger models and was a good performer on small models)
- random flip + rotate with small lighting, cropping and warping allowed (was only a good performer on smaller models but I wanted to keep it in the mix to help prevent overfitting on larger models)

producing 12 models per experiment.

The difference between Experiment 6 and 10 was that experiment 10 used early stopping/model saving as well as performed oversampling on the training set to help deal with class imbalance present in the dataset.

Ultimately, Experiment 10 didn't produce significantly different results in terms of a shift in accuracy but error analysis showed that the errors were distributed differently than the tests in Experiment 6.  As such, one of my two solutions was to ensemble (via averaging) all the trained models from these two runs (the submission that ultimately won).

## Experiment Details

[Experiment 1](1_-_initial_experiments.ipynb): Resnet (`resnet26d`) was fine-tuned on the dataset at a squished resolution of 128x128. Local accuracy with an 80/20 train/val split was just over 70%. Public leaderboard accuracy was 72.701% (rank 38).  Largely follows [Road to the Top, Part 1](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1).

[Experiment 2](2_-_small_models.ipynb): The submission is a fine-tuned ConvNeXt architecture (`convnext_small_in22k`) with 256x256 images, 12 epochs of training and using Test Time Augmentation (TTA). Had 79.2% local accuracy, 77.3% public leaderboard accuracy, current rank of 36.  Largely follows [Road to the Top, Part 2](https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2).

[Experiment 3](3_-_its_corn_-_failed_experiment_large_model.ipynb): Attempted to perform some experiments on a larger ConvNeXt architecture (`convnext_large_in22k`).  I was largely unhappy with the progress of those early thoughts and did not submit any results from these models.

[Experiment 4](4_-_fastai_ensemble.ipynb): Worked on an ensemble solution with Test Time Augmentation (TTA) following some fastai tutorials (with several tweaks). Generated a public leaderboard accuracy of 81.321% with a rank of 13th on the leaderboard (currently 69 participants).  Inspired by [Road to the Top, Part 3](https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3).

[Experiment 5](5_-_identify_performance_tweaks.ipynb): I started this notebook with the though of enumerating some of my thoughts at this stage to figure out "what next."  Ultimately, I mainly looked at one of the issues I thought may be affecting futhrer progress.  Namely, I was concerned that performing general batch augmentations might be too harsh and might be causing some inter-class confusion.  For example many differences in some of the corn kernel classifications come down to subtle colour changes and/or damage at the edge of the kernel (which could be effected by brightness/contrast adjustments or image cropping respectively).  I enumerated several sets of batch augmentations and ran them on some small model types (`resnet26d` and `convnext_small_in22k`).  I had hoped to load the saved models later in the notebook and do some error analysis but I had trouble recovering the test/train splits from the saved models (despite documentation suggesting it would be saved).  If anyone has any tips on that I'd love to hear it!  Ultimately, I identified 3 batch augmentation candidates that I would use moving forward:
  1. `flip_rotate`: flip and rotate randomly (simple, should be safe, rotate-only trained well on both models, rotate/flip was best performer on both models)
  1. `flip_rotate_smcrop`: flipping and rotating randomly with small cropping allowed
  1. `flip_rotate_smlightcropwarp`: flipping and rotating randomly with small lighting, cropping and affine warping changes allowed (figured it was best to keep this augmentation set alive to help potential overfitting when moving to larger models as it still performed well on small models though not top performer)

Fine tuned a large ConvNeXt model (`convnext_large_in22k`) on these three augmentation candidates and submitted the best (for fun).  The best single model from this test which had 71% ACC on public leaderboard.  Notably, it was one of my few submissions that had a significantly poorer performance on the public leaderboard than my 80/20 test-validation split had indicated (which was quite reliable throughout the competition).

[Experiment 6](6_-_large_model_ensemble_with_tested_augmentations.ipynb): Generated large-model ensemble with the pre-vetted augmentations from Experiment 5 and Test Time Augmentation (TTA).  Models included ConvNeXt (`convnext_large_in22k`), ViT (`vit_large_patch16_224`) and Swin (`swinv2_large_window12_192_22k`, `swin_large_patch4_window7_224`) architectures from the [Py**T**orch **IM**age **M**odels (TIMM) repository](https://github.com/rwightman/pytorch-image-models).  A main condition of selection was that they were pre-trained using [ImageNet](https://image-net.org/download) alone ([ruling out](https://github.com/rwightman/pytorch-image-models#pretrained-on-more-than-imagenet) some of their pre-trained models like EfficientNet), as per the competition rules.  This ensemble/augmentation combination provided, at the time, a personal-best submission on public leaderboard.  Accuracy of 81.6% with a rank of 19 on the public leaderboard (including submissions from higher ranking participants that have been deleted by organizer) at the time of submission. **The Kaggle inference-only submission of this work (which differed only by randomness induced by TTA) was my best score on the public leaderboard and was my runner-up submission on the private leaderboard).  The csv-only version of this work, despite having a lower public leaderboard score would have been my best submission on the private leaderboard achieving 82.429% accuracy.**

[Experiment 7](https://github.com/aspeers/pogchamps3/blob/main/7_-_simple_model_error_analysis.ipynb): Trained the same models as Experiment 6 and generated a confusion matrix to look at the distribution of errors in the validation sets.  As expected the classes most represented in the training data were over represented in the incorrect predictions. Many of the models get a disproportionate number of their incorrect predictions confusing discolored corn and predicting broken or pure instead. The model that seemed to do the best in this regard was `vit_large_patch16_224` with the   `flip_rotate_smlightcropwarp` augmentation.  I submitted this model (alone) to the leaderboard to see how it performed relative to the ensemble from Experiment 6.  This single model score had an accuracy of 77.729% on public leaderboard.  Would have loved to perform this experiment with the models saved in Experiment 6 but couldn't get it to recreate the same train/val sets (any suggestions are welcome).

[Experiment 8](8_-_testing_early_stopping_and_model_saving.ipynb):  Tested early stopping and model saving techniques in [fast.ai](https://www.fast.ai/).  Simply training all the models in the ensemble for a fixed number of epochs I knew was inefficient (at best) or allowing for some overfitting (at worst).  I wanted to implement early stopping to automatically stop training when the tracked parameter was no longer improving and model saving to keep the model when the tracked parameter was most recently improving but had never done this in the [fast.ai](https://www.fast.ai/) framework.  This notebook did some initial tests on using these techniques.  If I had more time, it would have been interesting to rerun the ensemble in Experiment 6 with these additions.

[Experiment 9](https://github.com/aspeers/pogchamps3/blob/main/9_-_oversampling_tests.ipynb): Ran tests on oversampling to help combat the class imbalance in the competition dataset.  As the dataset was unbalanced and the error analysis showed this could be causing some errors I attempted another method of combatting the problem and oversampled the minority classes when generating the training set.  Along with batch augmentation, I had hoped it would help boost my scores.  A reusable oversample class is produced with some initial tests.

[Experiment 10](10_-_early_stopping_oversampling_ensemble.ipynb): Producing an ensemble, similar to Experiment 6, where the models have the augmentation methods (from Experiment 5), early stopping / model saving (from Experiment 8) and oversampling (from Experiment 9). Confusion matrices showed that each class was now being the most likely prediction of the system although overall accuracy measures remain about the same on individual models. The ensemble provided a public leaderboard accuracy of 80.172% and a private leaderboard accuracy of 81.422%.

## Kaggle Competition Official Submissions
Two official submissions were made with Kaggle inference-only notebooks.

The winning submission combined the models from Experiment 6 and 10 in an averaged ensemble.  The inference-only notebook can be found [here](https://www.kaggle.com/code/andrewspeers/pogchamps3-experiment6plus10-kaggle-inference/notebook).

The second submission (which was higher on the public leaderboard) just included the models from Experiment 6.  This notebook can be found [here](https://www.kaggle.com/code/andrewspeers/pogchamps3-experiment6-kaggle-inference).

## Special Thanks
- I would like to thank 3x Kaggle Grandmaster [Rob Mulla](https://www.kaggle.com/robikscube) who hosted the competition and provided an excellent environment that encouraged participation.
- [Jeremy Howard](https://www.kaggle.com/jhoward) and [fast.ai](https://www.fast.ai/) whose courses and deep learning library were at the core of my progression throughout the competition.
- [Kurian Benoy](https://www.kaggle.com/kurianbenoy), a fellow Kaggle competitor, posted a [helpful discussion](https://www.kaggle.com/competitions/kaggle-pog-series-s01e03/discussion/353963) with relevant [fast.ai](https://www.fast.ai/) resources.  I had planned to use the [fast.ai](https://www.fast.ai/) library but was unaware of this Kaggle-centric walkthrough Jeremy Howard had done in the latest iteration of the course.  It was very encouraging and got me off to a good start.
- All the fellow competitors!  It was fun following along on everyone's discussion and progress throughout the two weeks.  While I'm still getting around to being more active in these communities it was great to see everyone engaged and it was contagious (in the good way :D).  Feel free to reach out and hope you found my walkthrough helpful!
