import time 
import network2 as net
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib
x_train, x_test, y_train, y_test = net.ml()

model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)
model.fit(x_train, y_train)
with open ('dump_random_forest.pkl', 'wb') as file :
    joblib.dump(model, file )


y_pred_dt = model.predict(x_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_dt)
precision = precision_score(y_test, y_pred_dt, average='macro')  # hoặc 'weighted' tùy vào bài toán
recall = recall_score(y_test, y_pred_dt, average='macro')
f1 = f1_score(y_test, y_pred_dt, average='macro')

print(f" Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Train acc:", model.score(x_train, y_train))
print("Test acc :", model.score(x_test, y_test))

# === ROC AUC cho từng label ===
# Lấy danh sách các lớp
classes = np.unique(y_test)

# Dự đoán xác suất
y_score = model.predict_proba(x_test)

# Binarize nhãn thật thành one-hot
y_test_bin = label_binarize(y_test, classes=classes)

# Tính ROC AUC cho từng lớp
print("\nROC AUC per label:")
for i, label in enumerate(classes):
    auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"Label {label} - ROC AUC: {auc:.4f}")
