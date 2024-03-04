# ECG Anomaly Detection

This is my thesis dedicated to detecting anomalies in the ECG of astronauts using neural networks. Its purpose is to assist doctors evaluating ECGs of astronauts at the JSC "TSNIImash." The program can quickly highlight segments of the ECG that the doctor should pay attention to (however, the program does not aim to replace the doctor's work, so less noticeable anomalies may be missed).

Warning: This project is created for JSC "TSNIImash", and training data and model files are confidential and not present in the repository. If you want to test the tools available here, you will need to do it with your own data.

Important: This is not the final product, and testing all functionality is currently impossible. However, you will find here instructions on what you can do now.

What is the goal of the final product? The program should take an ECG recording of arbitrary length as input and output the same recording with annotations in segments where anomalies are present. A segment is the interval between each pair of adjacent R-peaks in the ECG. Accordingly, if there is an anomaly in any of these segments, the program will mark that segment as anomalous.

A bit about the implementation: The project includes two neural networks. One of them identifies R-peaks in the ECG, allowing obtaining a set of sequential segments for evaluation from continuous data. The second neural network, in turn, binary classifies each obtained segment as containing anomalies or not. The project also includes graphical programs that facilitate data labeling for both neural networks. Additionally, there are two auxiliary programs for the first neural network: a data augmentator (increases the volume of training data without human effort) and a peak simplifier (mitigates some negative impacts in the operation of the first neural network).

Continuing with the practical use of the tools in the project:

1. To train the neural networks, you need to have one or several .txt files containing ECG of any length (the required file format will be shown at the end of the readme file).

2. First, you'll need to use the program peakMarker.py (in the Peak Marker folder) to mark the R-peaks in the files from which you want to train the first neural network to identify such peaks. After this step, you will obtain files storing the same data but in .parquet format, now with a column of labels for each point.

3. If you want to obtain more data for training the first neural network, you should pass your files to peakAugmentator.py (in the Peak Augmentator folder). This program will increase your training dataset by approximately 20 times and output files of augmented size, also in .parquet format.

4. Then, you need to perform training of the first neural network in the Peak Finder folder using the obtained data. Do this using the train.ipynb file.

5. Now you can automatically mark R-peaks in ECG using the trained neural network file in .pt (PyTorch) format and the nnFinder.py file. You can view the resulting marked file in .parquet format by opening it in Peak Marker. Here, you'll notice that the neural network often marks several points near each peak – to address this, you can use peakSimplifier.py (in the Peak Simplifier folder). Simply process your .parquet file with it and verify the changes by opening it again in Peak Marker.

6. Your next step will be to automatically mark files with ECG data from which you want to train the second neural network. After this, you need to mark each anomalous segment between R-peaks in these files. The sequenceMarker.py (in the Sequence Marker folder) will help you with this. When opening the file with it, if your automatically segmented files have sequences longer than one second, the program will color them orange (as it is an anomalous duration for the R-R segment in human ECG). Your own markings will be in red. Why? If the first neural network is not perfectly trained, it may not find all R-peaks, and consequently, the interval between the missed peaks on the left and right can be longer than a second. Therefore, it cannot be definitively stated that anomalies are present in segments longer than a second. This is why such segments are colored orange.

7. Once you finish marking the files for training the second neural network, you can proceed with training using the train.ipynb file in the Sequence Valuator folder. Then, you can test the neural network's performance by passing any ECG file with marked peaks to the nnValuator program. (Important: you need to process the file with marked peaks using the Sequence Marker program to color segments longer than a second in orange; otherwise, nnValuator will throw an error). The result of nnValuator.py is a Matplotlib animation showing how accurately the neural network was able to replicate segments from the input file. Since the second neural network is based on an autoencoder-like architecture, it will precisely replicate the original segments without anomalies and noticeably make errors on anomalous segments (if it is well-trained). This is how anomalies will be detected in the final product.

With this, the process of training the neural networks is completed. The final product will combine the functionality of all these tools in the following order:

1) Conversion of the input .txt file with the ECG under investigation into .parquet format.
2) Processing the .parquet file with the first neural network to segment the input ECG.
3) Identifying segments longer than 1 second in the file.
4) Evaluating and marking with the second neural network all segments shorter than one second.
5) Displaying the result - a file with the original ECG but in .parquet format and marked with anomalous segments (they will be red during visualization), as well as segments longer than a second (they will be orange).

To run the described programs on your computer, you need to have Python (I used 3.11), Jupyter Notebook and some third-party Python libraries, which you can install using the requirements.txt file (available in the repository): `pip install -r requirements.txt`

(Jupyter Notebook is required for .ipynb files. If you cannot use it, you can simply paste the code into a .py file or convert it using any method you are familiar with. However, it is worth mentioning that Jupyter Notebook is highly convenient for training neural networks).

Unfortunately, you will need to install the libxcb-cursor0 package globally.
Here is an example command for Debian: `sudo apt install libxcb-cursor0`

 You will also need cuda if you want to use the GPU when working with neural networks. You can find detailed instructions for installing cuda together with torch on the official PyTorch website: https://pytorch.org/get-started/locally/

Required format for .txt files:

```
Values (this is the header of the file; the program will skip it)
22.17742
22.5806459
22.9838717
23.3870975
23.7903233
23.3870975
23.7903233
24.1935491
25.0000008
23.7903233
22.5806459
21.7741942
21.3709684
20.5645168
```


This is an example of an ECG recording with 14 points (200 points = 1 second). There should be nothing else in your file. Alternatively, you can modify the parser in the peakMarker.py program.