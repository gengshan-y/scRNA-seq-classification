# scRNA-seq-classification

## REQUIREMENTS
- Tensorflow 1.0
- sklearn
- jupyter notebook

## PREPARE
- Use data_explore.ipynb to explore the data and preform split.
- Use build_data.ipynb to build training/val/test set.

## TRAIN
- Use main.py to train the mlp model
- Use train_svm.ipynb to train the SVM model
- Use vis_result.ipynb to monitor network training results.

## EVALUATION
- Use eval.py to evaluate the model
- I used voting among SVM, softmax MLP and sigmoid MLP to produce the final results, see build_eval.ipynb for details.

## RESOURCES
To accesss my data split, trained models and test predictions, see [this link](https://drive.google.com/file/d/1aTQIsOgEge06JVVUP6x3cXkXlE_L0tlq/view?usp=sharing).
