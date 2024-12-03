import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('imagens'):
    os.makedirs('imagens')

# Carregar o dataset
dataset_path = "breast.csv"
data = pd.read_csv(dataset_path)

# Separar as features (X) e os rótulos (y)
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values  

# Normalizar os dados para que todas as features tenham a mesma escala
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Configurar uma semente para reprodutibilidade
np.random.seed(42)

# Embaralhar os índices dos dados
indices = np.arange(len(y))
np.random.shuffle(indices)

# Aplicar o embaralhamento aos dados
X = X[indices]
y = y[indices]

# Divisão Holdout: 80% treino e 20% teste
train_size = int(0.8 * len(y))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Exibir informações sobre os conjuntos
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras\n")

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função de custo
def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradiente Descendente
def gradient_descent(X, y, w, b, alpha, epochs):
    m, n = X.shape
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        h = sigmoid(z)
        dw = np.dot(X.T, (h - y)) / m
        db = np.sum(h - y) / m
        w -= alpha * dw
        b -= alpha * db
    return w, b

# Funções de métrica
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score, tp, tn, fp, fn

# Função de validação cruzada em 10 folds
def k_fold_cross_validation(X, y, k=10, alpha=0.01, epochs=1000):
    fold_size = len(y) // k
    metrics = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # Inicializar parâmetros
        w = np.zeros(X_train.shape[1])
        b = 0

        # Treinar o modelo
        w, b = gradient_descent(X_train, y_train, w, b, alpha, epochs)

        # Fazer previsões no conjunto de teste
        z = np.dot(X_test, w) + b
        y_pred = (sigmoid(z) >= 0.5).astype(int)

        # Calcular métricas para o fold
        fold_metrics = calculate_metrics(y_test, y_pred)[:4]          metrics.append(fold_metrics)

    metrics = np.array(metrics)
    mean_metrics = np.mean(metrics, axis=0)
    std_metrics = np.std(metrics, axis=0)

    return mean_metrics, std_metrics

# Treinar o modelo com os dados de treino
alpha = 0.01  epochs = 1000  w = np.zeros(X_train.shape[1])  b = 0  
print("Treinando o modelo com holdout...\n")
w, b = gradient_descent(X_train, y_train, w, b, alpha, epochs)

# Fazer previsões no conjunto de teste
z_test = np.dot(X_test, w) + b
y_pred_test = (sigmoid(z_test) >= 0.5).astype(int)

# Calcular métricas e matriz de confusão
accuracy, precision, recall, f1_score, tp, tn, fp, fn = calculate_metrics(y_test, y_pred_test)

print("\nResultados no conjunto de teste (holdout):")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Revocação: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Exibir matriz de confusão
print("\nMatriz de Confusão:")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")

# Plotando os gráficos das métricas
metrics_names = ["Acurácia", "Precisão", "Revocação", "F1-Score"]
metrics_values = [accuracy, precision, recall, f1_score]
plt.figure(figsize=(8, 6))
plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
plt.title('Métricas de Avaliação do Modelo - Holdout')
plt.ylabel('Valor')
plt.savefig('imagens/metricas_holdout.png')
plt.show()

# Executando validação cruzada
print("\nExecutando validação cruzada em 10 folds...\n")
mean_metrics, std_metrics = k_fold_cross_validation(X, y, k=10, alpha=alpha, epochs=epochs)

# Exibir resultados da validação cruzada
print("\nResultados da validação cruzada (10 folds):")
for i, metric in enumerate(["Acurácia", "Precisão", "Revocação", "F1-Score"]):
    print(f"{metric}: Média = {mean_metrics[i]:.4f}, Desvio Padrão = {std_metrics[i]:.4f}")

# Plotando os gráficos das métricas da validação cruzada
plt.figure(figsize=(8, 6))
plt.bar(metrics_names, mean_metrics, yerr=std_metrics, capsize=5, color=['blue', 'green', 'orange', 'red'])
plt.title('Métricas de Avaliação do Modelo - Validação Cruzada')
plt.ylabel('Valor')
plt.savefig('imagens/metricas_kfold.png')
plt.show()
