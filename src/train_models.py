# coding: utf-8
"""
Model training and evaluation for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load processed data."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data: {self.df.shape}")
        return self.df
    
    def preprocess(self):
        """Preprocess features: encode categorical, scale numerical."""
        df = self.df.copy()
        
        # Target variable
        self.y = df['Churn'].map({'No': 0, 'Yes': 1})
        
        # Drop non-predictive columns
        drop_cols = ['customerID', 'Churn', 'tenure_group', 'total_charges_group']
        X = df.drop(columns=drop_cols, errors='ignore')
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Drop any NaN columns created during feature engineering
        X = X.fillna(X.mean(numeric_only=True))
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Preprocessed: Train={self.X_train.shape}, Test={self.X_test.shape}")
        print(f"  Churn distribution in train: {np.bincount(self.y_train)}")
        print(f"  Churn distribution in test: {np.bincount(self.y_test)}")
    
    def train_baseline_models(self):
        """Train Logistic Regression and Random Forest."""
        print("\nTraining Baseline Models...")
        
        # Logistic Regression
        print("  -> Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        self._evaluate_model('Logistic Regression', lr)
        
        # Random Forest
        print("  -> Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self._evaluate_model('Random Forest', rf)
    
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning."""
        print("\nTraining XGBoost...")
        
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        self._evaluate_model('XGBoost', xgb)
    
    def _evaluate_model(self, name, model):
        """Evaluate a model and store results."""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        self.results[name] = results
        
        print(f"    OK: {name}")
        print(f"      Accuracy: {results['Accuracy']:.4f}")
        print(f"      ROC-AUC: {results['ROC-AUC']:.4f}")
        print(f"      Recall: {results['Recall']:.4f}")
    
    def plot_evaluation(self, save_path=None):
        """Create comprehensive evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Evaluation - Churn Prediction', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison
        ax = axes[0, 0]
        metrics_df = pd.DataFrame({
            'Accuracy': [self.results[m]['Accuracy'] for m in self.models.keys()],
            'ROC-AUC': [self.results[m]['ROC-AUC'] for m in self.models.keys()]
        }, index=self.models.keys())
        metrics_df.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'])
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. ROC Curves
        ax = axes[0, 1]
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={self.results[name]['ROC-AUC']:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Confusion Matrix (Best Model)
        best_model_name = max(self.results, key=lambda x: self.results[x]['ROC-AUC'])
        ax = axes[1, 0]
        cm = self.results[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'Confusion Matrix - {best_model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # 4. Recall vs Precision
        ax = axes[1, 1]
        precision_vals = [self.results[m]['Precision'] for m in self.models.keys()]
        recall_vals = [self.results[m]['Recall'] for m in self.models.keys()]
        ax.scatter(recall_vals, precision_vals, s=200, alpha=0.6, c=range(len(self.models)), cmap='viridis')
        for i, name in enumerate(self.models.keys()):
            ax.annotate(name, (recall_vals[i], precision_vals[i]), fontsize=9, ha='center')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall Trade-off')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"OK: Evaluation plot saved to {save_path}")
        
        return fig
    
    def save_best_model(self, models_path):
        """Save the best performing model."""
        best_model_name = max(self.results, key=lambda x: self.results[x]['ROC-AUC'])
        best_model = self.models[best_model_name]
        
        model_file = Path(models_path) / f"churn_model_{best_model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(best_model, model_file)
        
        print(f"\nOK: Best model ({best_model_name}) saved to {model_file}")
        return best_model_name, model_file
    
    def generate_report(self):
        """Generate a summary report of model performance."""
        print("\n" + "="*60)
        print("CHURN PREDICTION MODEL EVALUATION REPORT")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {results['Accuracy']:.4f}")
            print(f"  Precision: {results['Precision']:.4f}")
            print(f"  Recall:    {results['Recall']:.4f}")
            print(f"  F1-Score:  {results['F1-Score']:.4f}")
            print(f"  ROC-AUC:   {results['ROC-AUC']:.4f}")
        
        print("\n" + "="*60)


if __name__ == '__main__':
    data_path = Path(__file__).parent.parent / "data" / "telco_churn_processed.csv"
    
    predictor = ChurnPredictor(data_path)
    predictor.load_data()
    predictor.preprocess()
    predictor.train_baseline_models()
    predictor.train_xgboost()
    predictor.plot_evaluation(save_path=Path(__file__).parent.parent / "models" / "evaluation.png")
    predictor.save_best_model(Path(__file__).parent.parent / "models")
    predictor.generate_report()
