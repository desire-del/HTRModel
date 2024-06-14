# Handwritten Text Recognition with TensorFlow


Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the RIMES off-line HTR dataset.
The model takes **images of text lines  as input** and **outputs the recognized text**.
3/4 of the words from the validation-set are correctly recognized, and the character error rate is around 20%.






## Command line arguments
* `--mode`: select between "train", "validate" and "infer". Defaults to "infer".
* `--decoder`: select from CTC decoders "bestpath", "beamsearch" and "wordbeamsearch". Defaults to "bestpath". For option "wordbeamsearch" see details below.
* `--batch_size`: batch size.
* `--data_dir`: directory containing the dataset.
* `--img_file`: image that is used for inference.
* `--dump`: dumps the output of the NN to CSV file(s) saved in the `dump` folder. Can be used as input for the [CTCDecoder](https://github.com/githubharald/CTCDecoder).


## Integrate word beam search decoding

The [word beam search decoder](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578) can be used instead of the two decoders shipped with TF.
Words are constrained to those contained in a dictionary, but arbitrary non-word character strings (numbers, punctuation marks) can still be recognized.
The following illustration shows a sample for which word beam search is able to recognize the correct text, while the other decoders fail.

![decoder_comparison](./res/decoder_comparison.png)

Follow these instructions to integrate word beam search decoding:

1. Clone repository [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)
2. Compile and install by running `pip install .` at the root level of the CTCWordBeamSearch repository
3. Specify the command line option `--decoder wordbeamsearch` when executing `main.py` to actually use the decoder



## Train model on RIMES dataset

### Prepare dataset
Follow these instructions to get the RIMES dataset:

* Download [RIMES DATASET](https://drive.google.com/drive/folders/1bhY1qccRPWn6pDI0XV7RZLKWbICQKqwt?usp=sharing)

### Run training

* Delete files from `model` directory if you want to train from scratch
* Go to the `src` directory and execute `python main.py --mode train --data_dir path/to/RIMES`
* The RIMES dataset is split into 95% training data and 5% validation data   
* Training stops after a fixed number of epochs without improvement

```
python main.py --mode train --data_dir path/to/rimes --batch_size 500 --early_stopping 15
```

## Information about model


It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.
For more details see this [Medium article](https://towardsdatascience.com/2326a3487cd5).


## References
* [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)
* [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)
* [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)
