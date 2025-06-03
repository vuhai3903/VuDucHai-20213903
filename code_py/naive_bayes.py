from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import network2 as net 
from sklearn.preprocessing import label_binarize
import numpy as np

x_train, x_test, y_train, y_test = net.ml()

# Tạo mô hình Categorical Naive Bayes
model = CategoricalNB()

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán
y_pred_dt = model.predict(x_test)

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