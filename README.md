# jubilant-octo-hmm
Hidden Markov Model with Numpy

POS tagging using Hidden Markov Model for multilingual corpora. <br>

Tested Tagging Accuracy: <br>
  Italian: 92% <br>
  Japanese: 89% <br>
  
To learn/map probabilities: <br>
```
python learn.py path/to/input ---> Generates vanillamodel.txt and averagedmodel.txt <br>
```
To tag raw data: <br>
```
python decode.py path/to/input --> Generates output.txt (contains tagged text data) <br>
```
To compare with tagged data: <br>
```
python compare.py path/to/input ----> Prints statistics to standard output
```
