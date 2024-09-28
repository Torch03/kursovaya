import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.datasets import load_files
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('IMDB Dataset.csv')
print(data.head())

X = data['review']
y = data['sentiment']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
print("Примеры токенов:", vectorizer.get_feature_names_out()[:10])


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

models = {
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'{name} F1-Score: {f1_score(y_test, y_pred):.4f}')
    print(f'{name} ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}')

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Ensemble Methods')
plt.legend()
plt.show()

# Важность признаков для Random Forest
best_rf = models['Random Forest']
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 6))
plt.title('Важность признаков для Random Forest')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [vectorizer.get_feature_names_out()[i] for i in indices])
plt.xlabel('Важность')
plt.show()

# Learning Curve для всех моделей
plt.figure(figsize=(10, 6))

for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label=f'{name} - Обучающая выборка')
    plt.plot(train_sizes, test_scores_mean, 'o-', label=f'{name} - Тестовая выборка')

plt.title('Learning Curve для моделей')
plt.xlabel('Количество обучающих примеров')
plt.ylabel('ROC-AUC')
plt.legend(loc='best')
plt.grid(True)
plt.show()