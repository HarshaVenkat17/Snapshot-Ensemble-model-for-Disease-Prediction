# Snapshot-Ensemble-model-for-Disease-Prediction
A snapshot ensemble model to create snapshots of ANN and use them to predict using ensemble technique. Uses Time-Based-Decay annealing for faster convergence than Cosine annealing.
1. Run "Create models.py" to create an ANN model using snapshot mechanism with Time-Based-Decay annealing and SGD optimizer to create models.
2. Run "Ensemble Predictions.py" to use these models to predict the output using Ensemble mechanism. Uses all the models created to predict values and performs "argmax" on them to find out the correct output. 
3. Test it on a single row and predicts the disease by applying inverse_transform in Label encoder module.
