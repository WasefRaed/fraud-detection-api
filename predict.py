import joblib
import pickle
import numpy as np
import pandas as pd

class FraudDetectionPipeline:

    def __init__(self):
        self.model     = joblib.load('models/model.pkl')
        self.explainer = joblib.load('models/explainer.pkl')
        self.scaler_amount = joblib.load('models/scaler_amount.pkl')
        self.scaler_time   = joblib.load('models/scaler_time.pkl')

        with open('models/features.pkl', 'rb') as f:
            self.features = pickle.load(f)

        with open('models/params.pkl', 'rb') as f:
            params = pickle.load(f)

        self.threshold = params['threshold']

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same feature engineering used in training."""
        out = df.copy()
        out['Amount'] = self.scaler_amount.transform(df[['Amount']])
        out['Time']   = self.scaler_time.transform(df[['Time']])

        out['Hour']             = (out['Time'] / 3600) % 24
        out['Amount_log']       = np.log1p(out['Amount'].clip(lower=0))
        out['V1_V2']            = out['V1'] * out['V2']
        out['V1_V3']            = out['V1'] * out['V3']
        out['Amount_deviation'] = out['Amount'] - out['Amount'].mean()
        out['V1_squared']       = out['V1'] ** 2
        out['Amount_Category']  = pd.cut(
            df['Amount'],
            bins=[-np.inf, 50, 100, 200, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)

        return out[self.features]

    def predict(self, transaction: dict) -> dict:
        df  = pd.DataFrame([transaction])
        X   = self._engineer(df)
        prob = float(self.model.predict_proba(X)[0, 1])
        flag = prob >= self.threshold

        if prob < 0.3:
            risk = 'Low'
        elif prob < 0.6:
            risk = 'Medium'
        elif prob < 0.8:
            risk = 'High'
        else:
            risk = 'Very High'

        # SHAP explanation
        shap_values = self.explainer.shap_values(X)
        shap_array  = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        
        shap_pairs = sorted(
            zip(list(X.columns), shap_array),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        top_features = [
            {
                "feature":   name,
                "impact":    round(float(val), 4),
                "direction": "increases fraud risk" if val > 0 else "decreases fraud risk"
            }
            for name, val in shap_pairs
        ]
        
        return {
            'is_fraud':          flag,
            'fraud_probability': round(prob * 100, 2),
            'risk_level':        risk,
            'recommendation':    'BLOCK TRANSACTION' if flag else 'ALLOW TRANSACTION',
            'top_reasons':       top_features
        }


# Single shared instance — loaded once when app starts
pipeline = FraudDetectionPipeline()