# textclassification for IMDB using CNN


how to run:
```
python3 imdb.py
```

output:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 500, 32)           3104      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8000)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 250)               2000250   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 251       
=================================================================
Total params: 2,163,605
Trainable params: 2,163,605
Non-trainable params: 0

Train on 25000 samples, validate on 25000 samples
Epoch 1/2
 - 13s - loss: 0.4612 - accuracy: 0.7530 - val_loss: 0.2870 - val_accuracy: 0.8811
Epoch 2/2
 - 12s - loss: 0.2144 - accuracy: 0.9152 - val_loss: 0.2657 - val_accuracy: 0.8892
Accuracy: 88.92%
```

reference:
https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/