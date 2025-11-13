"""
Phase 1: Simplified Model Training
Trains a basic Logistic Regression model for stock trend prediction.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Tuple

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


class ModelTrainerPhase1:
    """
    Phase 1: Simplified model training with Logistic Regression only.
    
    What this does:
    1. Prepares data for machine learning
    2. Trains a Logistic Regression model
    3. Evaluates performance
    4. Saves the model for Phase 2
    """

    def __init__(self):
        """Initialize the model trainer."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.accuracy = 0

        print("🤖 ModelTrainer Phase 1 initialized!")
        print("   Ready to train Logistic Regression model")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n🔧 Preparing data for machine learning...")
        print("=" * 45)

        # Remove rows with missing target
        df_clean = df.dropna(subset=['Target']).copy()

        # Define feature columns (exclude non-feature columns)
        exclude_cols = [
            'Date', 'Symbol', 'Target', 'Target_Binary', 'Next_Day_Return',
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]

        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

        # Prepare features and target
        X = df_clean[feature_cols].copy()
        y = df_clean['Target'].copy()

        print(f"📊 Dataset preparation:")
        print(f"   📈 Total samples: {len(X):,}")
        print(f"   🔧 Features: {len(feature_cols)}")
        print(f"   🎯 Target classes: {sorted(y.unique())}")

        # Show class distribution
        class_dist = y.value_counts().sort_index()
        print(f"   📊 Class distribution:")
        for class_val, count in class_dist.items():
            class_name = ['Down', 'Stable', 'Up'][class_val]
            print(f"      {class_name} ({class_val}): {count:,} ({count/len(y)*100:.1f}%)")

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("   🔧 Handling missing values...")
            X = X.fillna(X.median())

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Maintain class distribution
        )

        # Scale the features
        print("   📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n✅ Data preparation complete!")
        print(f"   🏋️  Training set: {len(X_train):,} samples")
        print(f"   🧪 Test set: {len(X_test):,} samples")

        # Store feature names
        self.feature_names = feature_cols

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print("\n🤖 Training Logistic Regression model...")
        print("=" * 40)

        # Initialize and train model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            multi_class='multinomial',
            solver='lbfgs'
        )

        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"✅ Model trained successfully!")
        print(f"   ⏱️  Training time: {training_time:.2f}s")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n📊 Evaluating model...")
        print("=" * 40)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.accuracy = accuracy

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }

        print(f"   📈 Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        print(f"\n   📊 Confusion Matrix:")
        print(f"      {'Predicted:':<15} {'Down':<10} {'Stable':<10} {'Up':<10}")
        print(f"      {'Actual:':<15} {'-'*30}")
        class_names = ['Down', 'Stable', 'Up']
        for i, row in enumerate(cm):
            print(f"      {class_names[i]:<15} {row[0]:<10} {row[1]:<10} {row[2]:<10}")

        print(f"\n   📋 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)

        return results

    def save_model(self, models_dir: str = "models") -> None:
        """
        Save trained model and metadata.
        
        Args:
            models_dir: Directory to save model
        """
        print(f"\n💾 Saving model to {models_dir}/...")

        # Create models directory
        os.makedirs(models_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(models_dir, "logistic_regression_phase1.pkl")
        joblib.dump(self.model, model_path)
        print(f"   ✅ Saved model to {model_path}")

        # Save scaler
        scaler_path = os.path.join(models_dir, "scaler_phase1.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✅ Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            'model_name': 'Logistic Regression',
            'model_type': 'Phase 1 - Basic Model',
            'accuracy': float(self.accuracy),
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'note': 'Phase 1: Basic model. Phase 2 will add multiple models and advanced features.'
        }

        metadata_path = os.path.join(models_dir, "model_metadata_phase1.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved metadata to {metadata_path}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            predictions, prediction_probabilities
        """
        if self.model is None:
            raise ValueError("No model available. Train a model first.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities


def main():
    """Example usage for Phase 1."""
    print("🚀 AI Market Trend Analysis - Phase 1 Model Training")
    print("=" * 60)

    try:
        # Load processed data
        print("📂 Loading processed stock data...")
        # Try Phase 1 features first, fallback to regular features
        try:
            df = pd.read_csv("data/features/stock_features_phase1.csv")
        except FileNotFoundError:
            df = pd.read_csv("data/features/stock_features.csv")
        
        df['Date'] = pd.to_datetime(df['Date'])

        print(f"   ✅ Loaded {len(df):,} rows of processed data")

        # Initialize trainer
        trainer = ModelTrainerPhase1()

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2)

        # Train model
        trainer.train_model(X_train, y_train)

        # Evaluate model
        results = trainer.evaluate_model(X_test, y_test)

        # Save model
        trainer.save_model()

        print("\n🎉 Phase 1 model training complete!")
        print("   📊 Model saved and ready for Phase 2 enhancements")
        print("\n   💡 Phase 2 will add:")
        print("      - Multiple ML models (Random Forest, XGBoost)")
        print("      - Cross-validation")
        print("      - Advanced metrics (F1, Precision, Recall)")
        print("      - Feature importance analysis")
        print("      - Model comparison")

    except FileNotFoundError:
        print("❌ Error: stock_features.csv not found!")
        print("   Please run feature_engineer_phase1.py first to create the processed features.")
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

