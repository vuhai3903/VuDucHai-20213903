import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import network2 as net 
import numpy as np

x_train, x_test, y_train, y_test = net.ml()


# ✅ Tối ưu KNN bằng GridSearchCV (tuỳ chọn)
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    metric='minkowski',
    metric_params=None,
    n_jobs=None
)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)



# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')



# In kết quả
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# === ROC AUC cho từng label ===
# Lấy danh sách các lớp
classes = np.unique(y_test)

# Dự đoán xác suất
y_score = knn.predict_proba(x_test)

# Binarize nhãn thật thành one-hot
y_test_bin = label_binarize(y_test, classes=classes)

# Tính ROC AUC cho từng lớp
print("\nROC AUC per label:")
for i, label in enumerate(classes):
    auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"Label {label} - ROC AUC: {auc:.4f}")
print("Train acc:",knn.score(x_train, y_train))
print("Test acc :",knn.score(x_test, y_test))