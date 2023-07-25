# BBC News Dataset Classifier

This repository contains code for a text classification model that classifies news articles from the BBC News dataset into different categories. The model is built using TensorFlow and Keras and uses an Embedding layer and GlobalAveragePooling1D layer for text representation.

## Dataset

The dataset used for training and evaluation is the BBC News dataset, which contains news articles categorized into different topics. The dataset consists of two main columns:

- `text`: Contains the news articles' text.
- `category`: Contains the corresponding category labels.

## Data Preprocessing

The data is preprocessed to prepare it for training. The following preprocessing steps are performed:

1. **Stopword Removal**: Common stopwords are removed from the news articles' text to reduce noise and improve the model's performance.

2. **Tokenization**: The text is tokenized, and the tokenizer is fit on the training set to convert the words into numeric sequences.

3. **Padding**: The tokenized sequences are padded to ensure uniform length, truncating or padding sentences to a maximum length of 120 words.

4. **Label Preprocessing**: The category labels are preprocessed to be compatible with the model. The labels are converted to sequences and then transformed into numerical arrays.

## Model Architecture

The text classification model architecture consists of the following layers:

1. **Embedding Layer**: The text data is first passed through an Embedding layer that maps words to dense vectors to learn word representations.

2. **GlobalAveragePooling1D Layer**: The outputs of the Embedding layer are averaged across all time steps to obtain a fixed-length output for each sample.

3. **Dense Layers**: The model then uses fully connected Dense layers with ReLU activation functions for feature extraction.

4. **Dropout Layer**: A Dropout layer is added to reduce overfitting during training.

5. **Output Layer**: The final layer uses a softmax activation function with 5 units (one for each category) for multi-class classification.

## Training

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. The training performance is visualized using plots for accuracy and loss over epochs.

## Results

The model is evaluated on the test set to measure its performance in classifying news articles into different categories. The accuracy and loss graphs are plotted to assess the model's training and validation performance.

Feel free to experiment with different hyperparameters, such as the number of epochs, batch size, or model architecture, to improve the model's accuracy further.

For any questions or suggestions, please contact [Francesco Alotto](mailto:franalotto94@gmail.com). Happy news classification with AI! 📰🤖
