Missile Reentry Phase Classifier

Project Overview:
This project implements a machine learning solution for classifying missile flight phases into reentry and non-reentry phases using telemetry data. The solution employs an LSTM neural network to process sequential data from missile tracks and follows MLOps best practices for deployment.

Features:
- LSTM-based classification of missile telemetry data
- Dockerized deployment for portability
- Kubernetes configuration for scalable deployment
- RESTful API for real-time predictions
- Logging of inference times and predictions

Data Description:
The model processes telemetry data from missile tracks, including:

Position coordinates (latitude, longitude, altitude)
Radiometric intensity
Sensor ID
Timestamps

Model Architecture:
The core of the solution is a multi-layer LSTM network that: 

Handles variable-length sequences of telemetry data
Uses dropout (0.3) and batch normalization for regularization
Employs early stopping to prevent overfitting
Achieves superior performance compared to a baseline altitude threshold classifier

Project Structure
mlops-missile-classifier/
├── api/                  # FastAPI implementation
│   └── app.py            # API endpoints and logic
├── kubernetes/           # Kubernetes deployment files
│   ├── deployment.yaml   # Deployment configuration
│   └── service.yaml      # Service configuration
├── models/               # Model definition and saved models
│   └── lstm_model.py     # LSTM implementation
├── src/                  # Core source code
│   ├── preprocessing.py  # Data preprocessing
│   └── evaluate.py       # Model evaluation
├── Dockerfile            # Docker configuration
├── requirements.txt      # Dependencies
├── run.py                # Training and inference script
└── README.md             # Documentation

Setup and Installation
Prerequisites

Python 3.9+
TensorFlow 2.8+
Docker
Kubernetes/Minikube (for deployment)

Local Development steps:

Clone the repository:
git clone https://github.com/shondon1/mlops-missile-classifier.git
cd mlops-missile-classifier

Screenshots:
![image](https://github.com/user-attachments/assets/a4c66e9b-7888-4b7b-8f8f-64926699f752)

## Resources that I used:
1. Claude AI
2. ChatGPT
3. https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python
4. https://github.com/rbhatia46/Data-Preprocessing-Template/blob/master/data_preprocessing_template.py
5. https://www.youtube.com/watch?v=9zhrxE5PQgY&ab_channel=SirajRaval
6. https://youtu.be/DQdB7wFEygo?si=ATKc-blQNfz7qEKw
7. https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
8. https://scikit-learn.org/stable/modules/model_evaluation.html
9. https://scikit-learn.org/stable/modules/preprocessing.html
10. https://scikit-learn.org/stable/modules/preprocessing.html
11. https://www.tensorflow.org/guide/checkpoint
12. https://www.tensorflow.org/guide/keras
13. https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning
14. https://www.datacamp.com/cheat-sheet/keras-cheat-sheet-neural-networks-in-python
15. https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/
16. https://www.youtube.com/watch?v=YCzL96nL7j0&ab_channel=StatQuestwithJoshStarmer

PS: I enjoy this challenge!! I continue to build on to this or make a new one similar to this