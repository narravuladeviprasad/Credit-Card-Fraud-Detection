import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, f1_score, precision_score, recall_score)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FraudDetectionPipeline:
    """Complete fraud detection pipeline with preprocessing, modeling, and evaluation."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        
    def load_data(self):
        """Load and preprocess dataset."""
        try:
            self.df = pd.read_csv(self.filepath, index_col=0)
            self.df.rename(columns={'is_fraud': 'Class', 'amt': 'Amount'}, inplace=True)
            
            # Select features (add more if available)
            self.features = ['Amount', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
            
            # Check if all features exist
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}. Using available features.")
                self.features = [f for f in self.features if f in self.df.columns]
            
            self.X = self.df[self.features]
            self.y = self.df['Class']
            
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.X.shape}")
            print(f"Features: {self.features}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def plot_class_distribution(self, y, title="Class Distribution"):
        """Visualize class distribution."""
        plt.figure(figsize=(8, 5))
        counts = y.value_counts()
        
        # Create bar plot
        bars = plt.bar(['Not Fraud (0)', 'Fraud (1)'], counts.values, 
                      color=['skyblue', 'salmon'], alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = len(y)
        fraud_pct = (counts[1] / total) * 100
        plt.text(0.5, max(counts) * 0.8, f'Fraud Rate: {fraud_pct:.2f}%', 
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.show()

    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== Data Exploration ===")
        print(f"Dataset Info:")
        print(f"Total samples: {len(self.df):,}")
        print(f"Features: {len(self.features)}")
        print(f"Missing values: {self.df[self.features].isnull().sum().sum()}")
        
        # Class distribution
        fraud_count = self.y.sum()
        total_count = len(self.y)
        print(f"\nClass Distribution:")
        print(f"Not Fraud: {total_count - fraud_count:,} ({((total_count - fraud_count)/total_count)*100:.2f}%)")
        print(f"Fraud: {fraud_count:,} ({(fraud_count/total_count)*100:.2f}%)")
        
        self.plot_class_distribution(self.y, "Original Class Distribution")
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        print(self.X.describe())

    def preprocess_data(self, X_train, y_train, sampling_strategy='undersampling'):
        """Scale and resample data."""
        # Scale numerical features
        numerical_features = ['Amount', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
        numerical_features = [f for f in numerical_features if f in X_train.columns]
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        
        # Apply sampling strategy
        if sampling_strategy == 'undersampling':
            sampler = RandomUnderSampler(random_state=42)
            print("Applying Random Under Sampling...")
        elif sampling_strategy == 'oversampling':
            sampler = SMOTE(random_state=42)
            print("Applying SMOTE Over Sampling...")
        else:
            return X_train_scaled, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
        
        print(f"After {sampling_strategy}:")
        print(f"Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled

    def initialize_models(self):
        """Initialize machine learning models with hyperparameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=5,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
        }

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation."""
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n=== {model_name} Results ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Fraud', 'Fraud'], 
                   yticklabels=['Not Fraud', 'Fraud'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()

        # ROC and PR curves (if probability predictions available)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            pr_auc = auc(recall_curve, precision_curve)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # ROC Curve
            ax1.plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC = {roc_auc:.3f}')
            ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
            ax1.set_xlabel('False Positive Rate', fontsize=11)
            ax1.set_ylabel('True Positive Rate', fontsize=11)
            ax1.set_title(f'{model_name} - ROC Curve', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # Precision-Recall Curve
            ax2.plot(recall_curve, precision_curve, color='green', linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
            ax2.set_xlabel('Recall', fontsize=11)
            ax2.set_ylabel('Precision', fontsize=11)
            ax2.set_title(f'{model_name} - Precision-Recall Curve', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
            
            return roc_auc, pr_auc, precision, recall, f1
        
        return None, None, precision, recall, f1

    def train_and_evaluate_models(self, sampling_strategy='undersampling'):
        """Train and evaluate all models."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.3, random_state=42
        )
        
        # Preprocess training data
        X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train, sampling_strategy)
        
        # Scale test data using the same scaler
        numerical_features = ['Amount', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
        numerical_features = [f for f in numerical_features if f in X_test.columns]
        
        X_test_scaled = X_test.copy()
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate each model
        self.results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_processed, y_train_processed)
            
            # Evaluate model
            roc_auc, pr_auc, precision, recall, f1 = self.evaluate_model(
                model, X_test_scaled, y_test, name
            )
            
            # Store results
            result = {
                'Model': name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC AUC': roc_auc if roc_auc else 'N/A',
                'PR AUC': pr_auc if pr_auc else 'N/A'
            }
            self.results.append(result)

    def display_results_summary(self):
        """Display comprehensive results summary."""
        if not self.results:
            print("No results to display. Train models first.")
            return
        
        # Create results DataFrame
        result_df = pd.DataFrame(self.results)
        
        # Sort by F1-Score (good balance of precision and recall for fraud detection)
        result_df = result_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(result_df.to_string(index=False, float_format='%.4f'))
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        # F1-Score comparison
        plt.subplot(2, 2, 1)
        bars = plt.bar(result_df['Model'], result_df['F1-Score'], color='lightcoral', alpha=0.7)
        plt.title('F1-Score Comparison', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('F1-Score')
        for bar, score in zip(bars, result_df['F1-Score']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Precision comparison
        plt.subplot(2, 2, 2)
        bars = plt.bar(result_df['Model'], result_df['Precision'], color='lightblue', alpha=0.7)
        plt.title('Precision Comparison', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Precision')
        for bar, score in zip(bars, result_df['Precision']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Recall comparison
        plt.subplot(2, 2, 3)
        bars = plt.bar(result_df['Model'], result_df['Recall'], color='lightgreen', alpha=0.7)
        plt.title('Recall Comparison', fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Recall')
        for bar, score in zip(bars, result_df['Recall']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # ROC AUC comparison (if available)
        if all(result_df['ROC AUC'] != 'N/A'):
            plt.subplot(2, 2, 4)
            roc_scores = [float(x) for x in result_df['ROC AUC']]
            bars = plt.bar(result_df['Model'], roc_scores, color='gold', alpha=0.7)
            plt.title('ROC AUC Comparison', fontweight='bold')
            plt.xticks(rotation=45)
            plt.ylabel('ROC AUC')
            for bar, score in zip(bars, roc_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Best model recommendation
        best_model = result_df.iloc[0]['Model']
        best_f1 = result_df.iloc[0]['F1-Score']
        print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
        print(f"   Best F1-Score: {best_f1:.4f}")
        print(f"   This model provides the best balance of precision and recall for fraud detection.")

    def run_pipeline(self, sampling_strategy='undersampling'):
        """Run the complete fraud detection pipeline."""
        print("="*60)
        print("FRAUD DETECTION PIPELINE")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Explore data
        self.explore_data()
        
        # Train and evaluate models
        print(f"\n{'='*60}")
        print("TRAINING AND EVALUATION")
        print(f"Sampling Strategy: {sampling_strategy.upper()}")
        print("="*60)
        
        self.train_and_evaluate_models(sampling_strategy)
        
        # Display results
        self.display_results_summary()
        
        return True


def main():
    """Main function to run fraud detection pipeline."""
    # Initialize pipeline
    filepath = 'fraudTest.csv'  # Update this path as needed
    pipeline = FraudDetectionPipeline(filepath)
    
    # Run pipeline with undersampling
    success = pipeline.run_pipeline(sampling_strategy='undersampling')
    
    if success:
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("Pipeline failed. Please check your data file and try again.")


if __name__ == '__main__':
    main()
