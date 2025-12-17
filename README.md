# 🔮 Customer Churn Prediction System

A comprehensive machine learning project to identify customers at risk of churning and provide actionable business insights. This internship project demonstrates end-to-end ML pipeline implementation with Telco customer data.

## 📋 Project Overview

**Goal**: Build a predictive model to identify which customers are likely to stop using a service, enabling proactive retention strategies.

**Dataset**: Telco Customer Churn (7,043 customers, 21 features)  
**Target**: Binary classification (Churned: Yes/No)  
**Models**: Logistic Regression, Random Forest, XGBoost  
**Churn Rate**: ~27%

## ✨ Key Features

- 📊 **Exploratory Data Analysis (EDA)** - Churn distribution, feature correlations
- 🔧 **Feature Engineering** - Tenure groups, charge ratios, service adoption scores
- 🤖 **Multiple Models** - Logistic Regression, Random Forest, XGBoost
- 📈 **Model Evaluation** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🎯 **Risk Segmentation** - Categorize customers into High/Medium/Low risk tiers
- 💡 **Business Insights** - Actionable recommendations for retention strategy
- 🌐 **Interactive Dashboard** - Streamlit app for real-time predictions

## 📁 Project Structure

```
Task-2/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── churn_prediction.ipynb    # Complete analysis notebook
│
├── src/
│   ├── data_prep.py         # Data loading, cleaning, feature engineering
│   └── train_models.py      # Model training and evaluation
│
├── app/
│   └── streamlit_app.py     # Interactive web dashboard
│
├── data/
│   ├── telco_churn.csv      # Raw dataset (auto-downloaded)
│   └── telco_churn_processed.csv  # Cleaned & engineered features
│
└── models/
    ├── churn_model_xgboost.pkl   # Best trained model
    └── evaluation.png            # Performance metrics chart
```

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
cd Task-2
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. **Prepare Data**
```bash
python src/data_prep.py
```

### 3. **Train Models**
```bash
python src/train_models.py
```

### 4. **Run Interactive Dashboard**
```bash
streamlit run app/streamlit_app.py
```

### 5. **Explore Notebook**
```bash
jupyter notebook churn_prediction.ipynb
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.80 | ~0.68 | ~0.53 | ~0.60 | ~0.84 |
| Random Forest | ~0.85 | ~0.72 | ~0.61 | ~0.66 | ~0.88 |
| **XGBoost** | **~0.86** | **~0.73** | **~0.65** | **~0.69** | **~0.89** |

**Winner**: XGBoost (best ROC-AUC score)

## 🎯 Key Churn Drivers

Top 5 features influencing churn predictions:
1. **Contract Type** - Month-to-month contracts have highest churn
2. **Tenure** - New customers (<6 months) churn more frequently
3. **Internet Service** - Fiber optic users show higher churn
4. **Monthly Charges** - Higher charges correlate with churn
5. **Tech Support** - Customers without support services churn more

## 💼 Business Insights

### Customer Risk Segments
- **High Risk** (27% of customers): Churn probability > 70%
- **Medium Risk** (31% of customers): Churn probability 40-70%
- **Low Risk** (42% of customers): Churn probability < 40%

### Recommendations
1. 🎯 **Targeted Retention** - Focus on high-risk customers with personalized offers
2. 📞 **Proactive Outreach** - Contact customers before churn occurs
3. 🔧 **Product Improvement** - Enhance support services and pricing flexibility
4. 📈 **Continuous Monitoring** - Retrain models quarterly with new data
5. 💰 **ROI Focus** - Retention cost typically << Acquisition cost

## 📊 Metrics Explained

- **Accuracy**: Overall prediction correctness (both classes)
- **Precision**: Of predicted churners, how many actually churn?
- **Recall**: Of all actual churners, how many did we catch?
- **F1-Score**: Harmonic mean of Precision & Recall
- **ROC-AUC**: Ability to distinguish between churn/no-churn across thresholds

## 🛠️ Technologies Used

| Component | Tools |
|-----------|-------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Notebook** | Jupyter |

## 📖 How to Use

### Running the Streamlit Dashboard
1. Access interactive predictions on test customers
2. Explore churn drivers by customer segment
3. View model performance metrics
4. Analyze key business insights

### Using the Jupyter Notebook
1. Step-by-step walkthrough of entire pipeline
2. Detailed EDA visualizations
3. Model training and evaluation
4. Feature importance analysis
5. Business recommendations

### Programmatic Usage
```python
from src.train_models import ChurnPredictor
from pathlib import Path

# Initialize predictor
predictor = ChurnPredictor('data/telco_churn_processed.csv')
predictor.load_data()
predictor.preprocess()
predictor.train_xgboost()

# Get predictions
churn_probability = predictor.models['XGBoost'].predict_proba(X_test)[0][1]
```

## 📚 Learning Outcomes

After completing this project, you'll understand:
- ✅ End-to-end ML pipeline: from data to production
- ✅ Classification model selection and evaluation
- ✅ Feature engineering for business problems
- ✅ Model comparison and hyperparameter tuning
- ✅ Risk segmentation and business impact analysis
- ✅ Translating ML results into actionable insights
- ✅ Building interactive dashboards for decision-makers

## 🔄 Future Enhancements

- [ ] Integrate real-time data pipeline
- [ ] Add SHAP values for individual prediction explanations
- [ ] Implement A/B testing framework for retention campaigns
- [ ] Deploy as REST API for production use
- [ ] Add customer lifetime value (CLV) predictions
- [ ] Create Power BI dashboard for executive reporting

## 📝 Notes

- **Data Download**: Dataset automatically downloads from GitHub if not present
- **Fallback**: If download fails, synthetic data is generated
- **Model Persistence**: Trained models saved to `models/` directory
- **Reproducibility**: All random seeds set to 42 for consistency

## 🤝 Contributing

Suggested improvements welcome! Consider:
- Trying alternative algorithms (LightGBM, CatBoost)
- Implementing ensemble methods
- Adding cross-validation analysis
- Creating additional business metrics

## 📞 Questions?

Refer to:
- 📓 `churn_prediction.ipynb` - Complete walkthrough
- 🔍 `src/data_prep.py` - Data handling details
- 🤖 `src/train_models.py` - Model implementation
- 🌐 `app/streamlit_app.py` - Dashboard code

---

**Status**: ✅ Complete  
**Last Updated**: December 2024  
**Version**: 1.0.0