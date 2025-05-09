from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import network2 as net 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np


x_train, x_test, y_train, y_test = net.ml()
# Khởi tạo SVM với kernel tuyến tính
svm = SVC(kernel='linear')

# Huấn luyện mô hình
svm.fit(x_train, y_train)

# Dự đoán
y_pred_svm = svm.predict(x_test)


# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_svm)
precision = precision_score(y_test, y_pred_svm, average='macro')  # hoặc 'weighted' tùy vào bài toán
recall = recall_score(y_test, y_pred_svm, average='macro')
f1 = f1_score(y_test, y_pred_svm, average='macro')

print(f"SVM Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")





# === ROC AUC cho từng label ===
# Lấy danh sách các lớp
classes = np.unique(y_test)

# Dự đoán xác suất
y_score = svm.predict_proba(x_test)

# Binarize nhãn thật thành one-hot
y_test_bin = label_binarize(y_test, classes=classes)

# Tính ROC AUC cho từng lớp
print("\nROC AUC per label:")
for i, label in enumerate(classes):
    auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"Label {label} - ROC AUC: {auc:.4f}")
