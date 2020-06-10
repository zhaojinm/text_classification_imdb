# textclassification for IMDB using RNN

requirement:
python3, numpy, keras2.3.1, tensorflow 1.13.1

how to run:
```python3 imdb.py```

output for GRU cell:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 500, 32)           160000    
_________________________________________________________________
gru_1 (GRU)                  (None, 32)                6240      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 33        
=================================================================
Total params: 166,273
Trainable params: 166,273
Non-trainable params: 0

Train on 25000 samples, validate on 25000 samples
Epoch 1/10
25000/25000 [==============================] - 92s 4ms/step - loss: 0.4415 - accuracy: 0.7865 - val_loss: 0.3594 - val_accuracy: 0.8458
Epoch 2/10
25000/25000 [==============================] - 93s 4ms/step - loss: 0.3011 - accuracy: 0.8774 - val_loss: 0.3308 - val_accuracy: 0.8616
Epoch 3/10
25000/25000 [==============================] - 92s 4ms/step - loss: 0.2599 - accuracy: 0.9010 - val_loss: 0.3282 - val_accuracy: 0.8674
Epoch 4/10
25000/25000 [==============================] - 92s 4ms/step - loss: 0.2238 - accuracy: 0.9161 - val_loss: 0.3218 - val_accuracy: 0.8726
Epoch 5/10
25000/25000 [==============================] - 92s 4ms/step - loss: 0.1925 - accuracy: 0.9310 - val_loss: 0.3307 - val_accuracy: 0.8731
Epoch 00005: early stopping
Accuracy: 87.31%
```




reference:
https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/