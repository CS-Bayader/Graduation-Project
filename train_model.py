import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# 1. تحميل البيانات
# ============================================================

df = pd.read_csv("diabetes.csv")

# ============================================================
# 2. اختيار المتغيرات الخمسة فقط
# ============================================================

features = [
    "Glucose",
    "Age",
    "BMI",
    "Insulin",
    "BloodPressure"
]

X = df[features]
y = df["Outcome"]

# ============================================================
# 3. معالجة القيم غير المنطقية/المفقودة
# ============================================================

# في هذه البيانات بعض الأعمدة قد تحتوي على 0
# رغم أن الصفر غير منطقي كقيمة طبية حقيقية.
# نحولها إلى NaN ثم نعوضها بالـ median.

columns_with_missing = [
    "Glucose",
    "BMI",
    "Insulin",
    "BloodPressure"
]

for column in columns_with_missing:
    X[column] = X[column].replace(0, pd.NA)
    X[column] = X[column].fillna(X[column].median())

# ============================================================
# 4. تقسيم البيانات إلى تدريب واختبار
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 5. Standardization
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. تدريب Logistic Regression
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================================
# 7. تقييم النموذج
# ============================================================

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_probability)

print("=" * 50)
print("Baseerah - Model Evaluation")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC: {auc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 8. حفظ النموذج والـ scaler
# ============================================================

joblib.dump(model, "baseerah_model.pkl")
joblib.dump(scaler, "baseerah_scaler.pkl")

# نحفظ أسماء المتغيرات أيضًا للتأكد من أن التطبيق
# يستخدم نفس الترتيب الذي استخدم أثناء التدريب.
joblib.dump(features, "baseerah_features.pkl")

print("\nModel saved successfully:")
print("baseerah_model.pkl")
print("baseerah_scaler.pkl")
print("baseerah_features.pkl")
