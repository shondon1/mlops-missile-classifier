# Purpose: Initial exploratory data analysis of missile telemetry data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder

def explore_data():
    """
    Perform initial exploratory data analysis on missile telemetry data.
    
    This function:
    1. Loads the training dataset
    2. Examines basic data structure and statistics
    3. Checks for missing values that might affect preprocessing
    4. Analyzes class distribution to assess balance/imbalance
    5. Visualizes key feature distributions
    """
    
    # Load the training data for exploration
    print("Loading training data...")
    train_df = pd.read_csv("data/train.csv")
    
    # Display the first few rows to understand data structure
    print("\n=== Sample Data ===")
    print(train_df.head())
    
    # Display dataset dimensions
    print("\n=== Dataset Size ===")
    print(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}")
    print(f"Dataset shape: {train_df.shape}")
    
    # Check for missing values that would require handling
    print("\n=== Missing Value Analysis ===")
    missing_values = train_df.isnull().sum()
    print(missing_values)
    missing_percentage = (missing_values / len(train_df)) * 100
    print(f"Missing value percentage: {missing_percentage}")
    
    # Analyze target variable distribution
    print("\n=== Class Distribution ===")
    class_counts = train_df['reentry_phase'].value_counts()
    print(class_counts)
    print(f"Class balance ratio: {class_counts[0]/class_counts[1]:.2f}:1")
    
    # Visualize key feature distributions
    print("\n=== Visualizing Altitude Distribution ===")
    plt.figure(figsize=(10, 6))
    train_df['altitude'].plot.hist(bins=50, alpha=0.7)
    plt.title('Altitude Distribution')
    plt.xlabel('Altitude')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Additional visualizations could be added here:
    # - Track duration histogram
    # - Radiometric intensity vs altitude
    # - Geographical distribution of tracks

if __name__ == "__main__":
    explore_data()