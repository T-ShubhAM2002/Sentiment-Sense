
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
        
    def backward_propagation(self, X, y, output, learning_rate):
        m = X.shape[0]
        
        # Calculate gradients
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, output, learning_rate)
            
            if i % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {i}, Loss: {loss}")
                
    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)
        
    def save_model(self, path):
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, path):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']

def preprocess_data(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Print columns to debug
    print("Columns in dataset:", df.columns.tolist())
    
    # Handle different possible label column names
    label_col = None
    possible_labels = ['Label', 'label', 'labels', 'category', 'class']
    
    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break
            
    if label_col is None:
        raise ValueError("No label column found in dataset. Expected one of: " + ", ".join(possible_labels))
    
    print(f"Using label column: '{label_col}'")
    
    # Convert labels to numerical values
    le = LabelEncoder()
    df['encoded_labels'] = le.fit_transform(df[label_col])
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Post'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else str(x))).toarray()
    y = pd.get_dummies(df['encoded_labels']).values
    
    return X, y, vectorizer, le

def train_ann_model():
    # Path to your dataset
    data_path = os.path.join('..', 'improved_dataset.csv')
    
    # Preprocess data
    X, y, vectorizer, le = preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train ANN
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = y_train.shape[1]
    
    ann = ANN(input_size, hidden_size, output_size)
    ann.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    
    # Save model and preprocessing objects
    os.makedirs('models', exist_ok=True)
    ann.save_model(os.path.join('models', 'ann_model.pkl'))
    pickle.dump(vectorizer, open(os.path.join('models', 'ann_vectorizer.pkl'), 'wb'))
    pickle.dump(le, open(os.path.join('models', 'ann_label_encoder.pkl'), 'wb'))
    
    # Evaluate
    predictions = ann.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_ann_model()