# ==========================================
# 1. IMPORTS
# ==========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ==========================================
# 2. CARREGAR OS DADOS
# ==========================================
dados = pd.read_csv("breast-cancer.csv")

print("Primeiras linhas do dataset:")
print(dados.head())


# ==========================================
# 3. TRATAMENTO DE DADOS
# ==========================================

# Troca valores "?" por NaN
dados = dados.replace("?", np.nan)

# Remove registros incompletos
dados = dados.dropna()

# Separa atributos e classe
dados_atributos = dados.drop(columns=["Class"])
dados_classes = dados["Class"]

# One-Hot Encoding (necessário porque o dataset é categórico)
dados_atributos = pd.get_dummies(dados_atributos)

print("\nAtributos após One-Hot Encoding:")
print(dados_atributos.head())


# ==========================================
# 4. HOLD OUT (70% treino / 30% teste)
# ==========================================
atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, random_state=42
)


# ==========================================
# 5. TREINAR O MODELO DECISION TREE
# ==========================================
tree = DecisionTreeClassifier()
bc_tree = tree.fit(atributos_train, classes_train)

print("\nClasses presentes no modelo:", bc_tree.classes_)


# ==========================================
# 6. PREVER TESTES
# ==========================================
predicoes = bc_tree.predict(atributos_test)

print("\nComparação entre classe real e prevista:")
for real, prev in zip(classes_test, predicoes):
    print(real, "-", prev)


# ==========================================
# 7. ACURÁCIA
# ==========================================
acuracia = metrics.accuracy_score(classes_test, predicoes)
print("\nAcurácia global:", acuracia)


# ==========================================
# 8. MATRIZ DE CONFUSÃO
# ==========================================
cm = confusion_matrix(classes_test, predicoes, labels=bc_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bc_tree.classes_)
disp.plot()
plt.show()

print("\nMatriz de Confusão (numérica):")
print(cm)


# ==========================================
# 9. CLASSIFICAR NOVA INSTÂNCIA
# ==========================================

# Exemplo fictício (os valores precisam seguir o formato original do dataset)
nova_instancia = {
    "age": ["40-49"],
    "menopause": ["premeno"],
    "tumor-size": ["20-24"],
    "inv-nodes": ["0-2"],
    "node-caps": ["no"],
    "deg-malig": [2],
    "breast": ["left"],
    "breast-quad": ["left_up"],
    "irradiat": ["no"]
}

# Transforma em DataFrame
nova_instancia_df = pd.DataFrame(nova_instancia)

# One-Hot Encoding com alinhamento das colunas
nova_instancia_df = pd.get_dummies(nova_instancia_df)
nova_instancia_df = nova_instancia_df.reindex(columns=dados_atributos.columns, fill_value=0)

classe_prevista = bc_tree.predict(nova_instancia_df)

print("\nClasse prevista para nova instância:", classe_prevista)
