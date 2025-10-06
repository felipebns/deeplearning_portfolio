# Classificação - Satisfação de Passageiro em Linhas Aéreas

## Autores
- Felipe Bakowski Nantes de Souza  
- Vinicius Grecco Fonseca Mulato  
- Victor Soares

### Nota: 
- O cógido foi feito considerando um arquivo auxiliar mlp e um outro utils. Aquele possui a implementação completa da mlp e este funções auxiliares que ajudam na organização

# 1.    Data set - Seleção

## Data set escolhido: Análise de satisfação de passageiros em voos, Kaggle

Esse data set devido a uma soma de alguns fatores. De início, ele representa um desafio real que muitas empresas precisam resolver e é interessante interagir com um problema que tenham reflexões reais. Ainda, ele possui grande número de linhas, proporcionando uma análise mais robusta, muitas features de todos os tipos e uma variável target bem balanceada. Criando assim, um terreno fértil para a análise de dados.

- Fonte: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

# 2. Data set - Explicação

Esse data set possui 25 colunas, 24 sendo potenciais features e 1 target (satisfação). Ela é uma variável categórica não ordenada e assume 2 valores: satisfeito ou neutro/insatisfeito. Ainda, a target está bem balanceada, estando divida em 43/57 %

Em relação as features, elas variam entre qualitativas e quantitativas, sendo majoritariamente qualitativas e com poucas linhas com valores faltando.

### Colunas

Agora, pode-se observar um código que concatena as bases de treino e teste, visualiza o número de linhas totais e cita todas as colunas que possuem no dataframe

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlp import mlp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import plot_distributions

# carregar os dois conjuntos
df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")

# marcar a origem para poder separar depois
df_train["_source"] = "train"
df_test["_source"]  = "test"

# concatenar
df = pd.concat([df_train, df_test], ignore_index=True)

print("Shape combinado:", df.shape)
print(df["_source"].value_counts())

target = df['satisfaction']

df.columns
```
### Output
```
Shape combinado: (129880, 26)
_source
train    103904
test      25976
Name: count, dtype: int64
Index(['Unnamed: 0', 'id', 'Gender', 'Customer Type', 'Age', 'Type of Travel',
       'Class', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'satisfaction', '_source'],
      dtype='object')
```

### Visualizando equilíbrio na variável Target (satisfaction)

```python
plt.figure(figsize=(6,4))
ax = sns.histplot(x=target, stat="percent", discrete=True)

plt.title("Distribuição da variável 'satisfaction' (%)")
plt.xlabel("Satisfação")
plt.ylabel("Porcentagem")
plt.show()
```

![equilibrio](equilibrio.png)

### Preenchendo valores vazios

```python
print(df.columns[df.isnull().any()]) #observa-se colunas com valores faltando!
```

```
Index(['Arrival Delay in Minutes'], dtype='object')
```

### Arrival delay é numérica, utilizaremos a moda para preencher o valor, já que se arrival delay está como null, provavelmente foi 0 e esqueceram de colocar

```python
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mode()[0])
print(df.columns[df.isnull().any()]) #roda mais uma vez para garantir
```

```
Index([], dtype='object')
```

### Visualização features

```python
# definir colunas
quantitative_cols = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

ordinal_cols = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

nominal_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
target_col = "satisfaction"

# -------- PLOT ORIGINAL --------
plot_distributions(df, quantitative_cols, ordinal_cols, nominal_cols, title_suffix="(Original)", force_ordinal_continuous=False)
```

![quant_semnorm](quant_semnorm.png)
![ord_semnorm](ord_semnorm.png)
![nominais_semnorm](nominais_semnorm.png)

Pelas distribuições apresentadas, as variáveis categóricas estão relativamente balanceadas em alguns aspectos, mas apresentam diferenças relevantes em outros. O gênero está equilibrado entre homens e mulheres. Já o tipo de cliente é bastante desbalanceado, com predominância de clientes leais. No tipo de viagem, há mais viagens de negócios do que pessoais. Em relação à classe, as categorias Business e Eco têm proporções próximas, enquanto Eco Plus aparece em bem menor quantidade. A variável de satisfação também é relativamente balanceada, com uma leve maioria de clientes insatisfeitos ou neutros.

Nas variáveis numéricas, observa-se diversidade: a idade segue uma distribuição concentrada entre 20 e 50 anos; a distância do voo é enviesada para valores menores; e os atrasos de partida e chegada apresentam forte concentração em atrasos curtos, com alguns outliers de longos atrasos. Já os serviços avaliativos (como wifi, comida, embarque, conforto de assento, limpeza, etc.) mostram distribuições variadas, mas tendem a concentrar respostas em notas intermediárias a altas, o que sugere certo viés positivo nas avaliações.

# 3. Limpeza de dados e normalização

###  Z-Score (média 0, desvio 1)

Usamos em **Age** e nas variáveis **ordinais (avaliações de 1 a 5)**.

* **Por quê?**

  * O Z-score centraliza os dados na média e escala pela variabilidade.
  * Isso coloca todas essas variáveis em uma escala comparável (valores entre -2 e 2, geralmente).
  * É útil quando os dados são aproximadamente simétricos ou queremos destacar desvios em relação à média.

Exemplo: Idades diferentes são comparadas em termos de "quantos desvios padrão acima ou abaixo da média" estão.


###  Min-Max [-1, 1]

Usamos em **Flight Distance** e nos **delays (após log)**.

* **Por quê?**

  * O Min-Max traz os valores para um intervalo fixo, aqui entre -1 e 1.
  * Isso garante que nenhuma variável tenha escala muito maior que as outras.
  * É útil quando a distribuição não é centrada na média, mas queremos que o modelo “veja” tudo na mesma faixa.


###  Log + Min-Max

Usamos em **Departure Delay** e **Arrival Delay**.

* **Por quê?**

  * Atrasos têm distribuição muito enviesada: muitos voos com atraso 0 ou baixo, e poucos voos com atrasos enormes.
  * O log “comprime” esses valores grandes, reduzindo o impacto dos extremos.
  * Depois, aplicamos Min-Max para trazer o resultado para a faixa [-1,1], alinhando com as outras features.


###  One-Hot Encoding (nominais)

Nas variáveis como **Gender, Customer Type, Type of Travel, Class**.

* **Por quê?**

  * São categorias sem ordem (ex.: “Male” ≠ maior que “Female”).
  * O One-Hot cria colunas binárias (`0` ou `1`) para cada categoria, sem necessidade de normalização extra.

## Implementando a normalização

```python
# -------- ONE-HOT + TARGET --------
df_encoded = pd.get_dummies(df, columns=nominal_cols, dtype=int)
df_encoded[target_col] = df_encoded[target_col].map({
    "satisfied": 1,
    "neutral or dissatisfied": 0
})

# -------- NORMALIZAÇÃO --------
df_norm = df_encoded.copy()

# Age + ordinais -> Z-score
scaler_z = StandardScaler()
df_norm[["Age"] + ordinal_cols] = scaler_z.fit_transform(df_norm[["Age"] + ordinal_cols])

# Flight Distance -> MinMax [-1,1]
scaler_fd = MinMaxScaler(feature_range=(-1,1))
df_norm["Flight Distance"] = scaler_fd.fit_transform(df_norm[["Flight Distance"]])

# Delays -> log + MinMax [-1,1]
for col in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    df_norm[col] = np.log1p(df_norm[col].clip(lower=0))
    scaler_delay = MinMaxScaler(feature_range=(-1,1))
    df_norm[col] = scaler_delay.fit_transform(df_norm[[col]])

df_norm.head()
# -------- PLOT NORMALIZADO --------
plot_distributions(
    df_norm,
    quantitative_cols,
    ordinal_cols,
    [],  # <- passa lista vazia, não plota nominais
    title_suffix="(Normalizado)",
    force_ordinal_continuous=True
)
```

![quant_comnorm](quant_comnorm.png)
![ord_comnorm](ord_comnorm.png)

# 4. Implementação MLP

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class mlp:
    def __init__(self, n_features:int, n_hidden_layers: int, n_neurons_per_layer: list, 
                activation: str, loss: str, optimizer: str, epochs: int, eta: float) -> None:
        self.n_features = n_features
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.eta = eta

        self.weights = []
        self.biases = []
        self.layer_dims = [n_features] + n_neurons_per_layer

        for i in range(len(self.layer_dims) - 1):
            w = np.random.randn(self.layer_dims[i+1], self.layer_dims[i]) * 0.1
            b = np.zeros((self.layer_dims[i+1],))
            self.weights.append(w)
            self.biases.append(b)

        # histórico de treino
        self.history = {"loss": [], "accuracy": []}

        print("\n=== Inicialização de Pesos e Biases ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Camada {i+1}:")
            print(f"W{i+1} shape {w.shape}:\n{w}")
            print(f"b{i+1} shape {b.shape}:\n{b}\n")

    def train(self, X, y, threshold: float = 5e-3, window: int = 10) -> None:
        loss_history = []

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(len(y)):
                y_pred, cache = self.forward_pass(X[i])
                loss = self.loss_calculation(y[i], y_pred)
                total_loss += loss
                grads_w, grads_b = self.backpropagation(y[i], y_pred, cache)
                self.update_parameters(grads_w, grads_b)

            avg_loss = total_loss / len(y)
            loss_history.append(avg_loss)

            preds_train = self.test(X)
            train_acc = self.calculate_accuracy(y, preds_train)

            # armazenar no histórico
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(train_acc)

            if epoch % 10 == 0:
                acc_str = f"{train_acc*100:.2f}%" if not np.isnan(train_acc) else "nan"
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Train Acc: {acc_str}")

            # critério de parada: média móvel dos últimos "window" epochs
            if epoch >= window:
                moving_avg_prev = np.mean(loss_history[-2*window:-window])
                moving_avg_curr = np.mean(loss_history[-window:])
                if abs(moving_avg_prev - moving_avg_curr) < threshold:
                    print(f"Treinamento encerrado no epoch {epoch} (convergência detectada).")
                    break

    def test(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(len(X)): 
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                preds.append(np.argmax(y_pred))
            else:
                val = np.array(y_pred).ravel()
                v = val[0] if val.size > 0 else val
                preds.append(1 if v > 0.5 else 0)
        return np.array(preds)


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_confusion: bool, plot_roc: bool, preds: np.ndarray) -> None:
        acc = self.calculate_accuracy(y, preds)
        print(f"Accuracy: {acc*100:.2f}%")

        print("\n=== Pesos e Biases do Modelo ===")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"\nCamada {i+1}:")
            print(f"  Pesos W{i+1} (shape {w.shape}):")
            print(w)
            print(f"  Biases b{i+1} (shape {b.shape}):")
            print(b)

        binary = (len(np.unique(y)) == 2)

        if plot_confusion:
            self.plot_confusion_matrix(y, preds)

        if plot_roc and binary:
            self.plot_roc_curve(X, y)

        # sempre plota histórico se existir
        if self.history and len(self.history.get("loss", [])) > 0:
            self.plot_history()

    # -------- Funções auxiliares de plot --------
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=np.unique(y_true),
                    yticklabels=np.unique(y_true))
        plt.title("Confusion Matrix (normalized by row)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_roc_curve(self, X, y_true):
        y_scores = []
        for i in range(len(X)):
            y_pred, _ = self.forward_pass(X[i])
            if self.loss == "cross_entropy":
                y_scores.append(y_pred[1])
            else:
                val = np.array(y_pred).ravel()
                y_scores.append(val[0] if val.size > 0 else val)
        y_scores = np.array(y_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()

    def plot_history(self):
        """Plota loss e accuracy armazenados em self.history."""
        loss = self.history.get("loss", [])
        acc = self.history.get("accuracy", [])
        epochs = np.arange(1, len(loss)+1)

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        # loss
        axes[0].plot(epochs, loss, marker="o")
        axes[0].set_title("Loss por Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        # accuracy
        axes[1].plot(epochs, acc, marker="o")
        axes[1].set_title("Accuracy por Epoch (treino)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    def forward_pass(self, x: np.ndarray) -> tuple:
        a = x
        cache = {"z": [], "a": [a]}
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            if i == len(self.weights) - 1 and self.loss == "cross_entropy":
                a = self.softmax(z)
            else:
                a = self.activation_function(z)
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    def backpropagation(self, y_true: np.ndarray, y_pred: np.ndarray, cache: dict) -> tuple:
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Última camada
        if self.loss == "cross_entropy":
            delta = self.derive_cross_entropy(y_true, y_pred)
        elif self.loss == "mse":
            dloss_dy_pred = self.derive_mse(y_true, y_pred)
            if self.activation == "sigmoid":
                delta = dloss_dy_pred * self.derive_sigmoid(cache["z"][-1])
            elif self.activation == "tanh":
                delta = dloss_dy_pred * self.derive_tanh(cache["z"][-1])
            elif self.activation == "relu":
                delta = dloss_dy_pred * self.derive_relu(cache["z"][-1])
        else:
            raise ValueError("Loss não suportada")

        grads_w[-1] = np.outer(delta, cache["a"][-2])
        grads_b[-1] = delta

        # Camadas ocultas
        for l in reversed(range(len(self.weights)-1)):
            delta = np.dot(self.weights[l+1].T, delta)
            if self.activation == "sigmoid":
                delta *= self.derive_sigmoid(cache["z"][l])
            elif self.activation == "tanh":
                delta *= self.derive_tanh(cache["z"][l])
            elif self.activation == "relu":
                delta *= self.derive_relu(cache["z"][l])
            grads_w[l] = np.outer(delta, cache["a"][l])
            grads_b[l] = delta

        return grads_w, grads_b
        
    def update_parameters(self, grads_w, grads_b):
        if self.optimizer == "gd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.eta * grads_w[i]
                self.biases[i]  -= self.eta * grads_b[i]
        else:
            raise ValueError(f"Optimizer {self.optimizer} não suportado")

    def loss_calculation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.loss == 'mse':
            return self.mse(y_true, y_pred)
        elif self.loss == 'cross_entropy':
            return self.cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Função de loss {self.loss} não suportada")

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'relu':
            return self.relu(z)
        else:
            raise ValueError(f"Função de ativação {self.activation} não suportada")
        
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

    def derive_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2*(y_true - y_pred)

    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true_onehot * np.log(y_pred))

    def derive_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        num_classes = len(y_pred)
        y_true_onehot = np.eye(num_classes)[y_true]
        return y_pred - y_true_onehot

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def derive_sigmoid(self, z: np.ndarray) -> np.ndarray:
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derive_tanh(self, z: np.ndarray) -> np.ndarray:
        return 1 - (np.tanh(z))**2

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derive_relu(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
```

### Observe
Essa implementação de MLP permite que o usuário escolha o número de layers e seu tamanho (output também), permite que o usuário escolhe entre Relu, sigmoid ou tanh como função de ativação, disponibiliza cross-entropy ou MSE para calculo de erro, mas tem a limitação de que só permite o usuário escolher o Gradient Descent padrão (GD) como otimização, mas como ele trabalha em modo estocástico (atualiza peso em cada amostra), pode-se considerar um SGD.

Por fim, existem alguns parâmetros chaves que podem ser alterados na chamada da classe que são cruciais para o funcionamento da rede neural. Eles são: learning rate ('eta' no código), epochs e o número de camadas ocultas, assim como o número de neurônios nela.

# 5. Treinando Modelo

Assim, temos a nossa MLP pronta, agora basta implementar, para isso utilizamos os seguintes parâmetros:

```python
model = mlp(
    n_features=X_train.shape[1],
    n_hidden_layers=2,
    n_neurons_per_layer=[32, 16, 2],
    activation="relu",
    loss="mse",
    optimizer="gd",
    epochs=100,
    eta=0.01
)

model.train(X_train, y_train)
```
Quando fazemos model = mlp(...) os pesos e biases já são inicializados automaticamente:

```
=== Inicialização de Pesos e Biases ===
Camada 1:
W1 shape (32, 27):
[[ 1.15933603e-01 -1.81859311e-02  3.78720797e-02  1.61887445e-01
  -1.76264976e-02 -1.17896559e-01  2.29659597e-01  8.47392716e-02
   7.84833501e-02  1.17116376e-01 -3.29254012e-02  1.47752258e-01
  -5.05408257e-02 -1.03760407e-02 -6.75068207e-02  7.65876531e-02
   2.24685014e-01 -4.85771329e-02  9.20202828e-02 -1.39268909e-01
  -1.33149834e-01  7.09939469e-02  9.62322284e-03  1.97430345e-01
   7.96567200e-02 -2.02657840e-01  9.01385734e-03]
 [-6.04410809e-02  1.55657322e-02 -6.76827315e-02  8.16054968e-02
   9.60520887e-03 -1.25204605e-01 -6.19064575e-02 -6.01804224e-02
   5.45090162e-02  2.06770711e-01  8.75548772e-02  7.76242877e-02
   9.78846571e-03 -3.32076578e-02 -2.81371297e-02 -1.04293824e-01
  -2.11831322e-01  9.67483947e-02  1.19716602e-02  4.71019153e-02
   8.49512245e-03 -9.58052260e-02  1.06920949e-01  1.34581558e-01
   1.10775335e-01 -1.21445693e-02 -8.24293321e-03]
 [-1.07644203e-01  1.19474816e-01 -1.23425620e-02 -4.76688842e-02
  -1.30960013e-01  8.46369514e-02 -2.03250433e-02 -4.67880145e-02
  -3.55089096e-02  1.30250049e-01  1.03907897e-02  1.98612609e-01
  -9.29014497e-02 -3.78056341e-02  9.92537078e-02  6.86785045e-02
   1.28081217e-01  3.12844113e-02 -1.04421485e-01 -1.45493638e-01
   2.50604195e-01 -1.04281028e-01  1.41494095e-02 -1.53304959e-02
  -2.31537260e-01 -5.91761023e-03 -6.21277588e-02]
...
   0.09038222 -0.10695692 -0.02021934 -0.06621559]]
b3 shape (2,):
[0. 0.]
```

Observe que escolhemos cross_entropy como função de cálculo de loss, devido a isso, devemos ter um output de 2 neurônios, cada um deles diz a chance de uma amostra pertencer a uma classe. Mas a MLP formata isso de modo que no final o resultado fica apenas 1 ou 0 para comparação nos testes

Quando usamos model.train, a mlp já toma conta do treinamento sozinho, fazendo todo o processo de forward propagation, loss calculation, backpropagation e atualização de parâmetros. O único problema que encontrei no treinamento foi uma demora excessiva na convergência, isso pode ser explicado pelo eta baixo que foi escolhido. Mas, a solução encontrada foi atualizar o threshold de parada para um valor menor, sem prejudicar a qualidade do modelo.

# 6. Estratégia de treino e teste

O dataset foi extraído do Kaggle já em 2 seções, treino e teste (75/25), como pode ser observado na primeira seção desse documento

Em relação ao modo de treino foi feito um treinamento em modo estocástico, ou seja, atualiza os pesos em cada amostra. Isso é útil para esse caso, pois esse é o método mais rápido para datasets grandes (100K+ linhas).

Ainda, é implementado um método de early stopping para evitir o overfitting, basicamente, analisamos o loss médio de uma janela (parâmetro ajustável) de epochs e verificamos se a o loss atual menos o loss médio dessa janela é menor que um threshold de parada, caso sim, isso indica convergência. Tendo isso em vista, pode-se para o modelo já para evitar o overfitting.

Convergência:

```
Epoch 0, Loss: 0.162550, Train Acc: 94.72%
Epoch 10, Loss: 0.093688, Train Acc: 96.05%
Epoch 20, Loss: 0.088143, Train Acc: 96.25%
Treinamento encerrado no epoch 27 (convergência detectada).
```

### Implementação do teste:
```python
preds_test = model.test(X_test)
```

# VALIDATION ????????

# 7. Curva de erro e visualização

### Extraindo dados para avaliar modelo:

```python
model.evaluate(X_test, y_test, plot_confusion=True, plot_roc=True, preds=preds_test)
```

## Resultado:

```
Accuracy: 96.01%

=== Pesos e Biases do Modelo ===

Camada 1:
  Pesos W1 (shape (32, 27)):
[[-5.90536709e-01  4.66615554e-01 -4.49108095e-01  5.21877533e-03
   3.25350763e-01  7.17445084e-02 -1.22360259e-02  1.18032302e+00
   3.48788226e+00 -1.01996644e-02  4.23905953e-01  2.89380884e-02
  -1.68034617e+00 -1.03658232e-01  4.08442980e-01  7.29463962e-01
   8.41765767e-02  1.99656636e-01 -3.52814596e-01 -6.00852790e-01
   3.88832152e-01 -1.35740680e+00 -2.79580057e-01 -4.19785134e-01
   2.84328537e-01 -9.54605775e-01 -3.50128783e-01]
 [ 2.11480807e-01  1.00159878e-01 -3.06430613e-01  1.81622933e-01
  -1.74595611e-01  7.60054834e-01 -1.08294362e-01 -6.50388801e-01
  -1.00690675e+00  1.99061008e+00  3.52784516e-01  6.80065608e-01
   2.42878895e-01 -4.88982385e-02  4.75156172e-01  1.12191729e-01
  -4.96634271e-01  2.57593430e-01 -3.83644338e-01 -2.87492645e-01
   5.30848775e-02 -8.70605540e-01 -3.47258423e-01 -1.41449629e-01
   8.23021086e-01 -1.08758530e+00 -3.75258515e-01]
 [-2.07872079e+00 -1.46026838e-01 -1.58100799e-02 -1.47463794e-01
  -3.72018818e-02  2.47253169e-01  4.14674460e-01 -1.79369636e-01
  -1.87667054e-01  5.79683551e-01 -3.35868181e-01 -3.13726939e-02
  -6.65274745e-01 -4.75407953e-01 -2.27500091e-01  1.13320093e+00
   4.41707412e-01 -3.28317828e-02 -6.19415108e-01 -6.24369083e-01
...
   0.25893154  0.18075803  0.69442195 -0.29597184 -0.38280426 -0.71957644
   0.460514   -0.63309972 -0.2926815  -0.44381686]]
  Biases b3 (shape (2,)):
[ 0.46715727 -0.46715727]
```

## Gráficos:

### Matriz de confusão:

![confusao](confusao.png)

### Curva ROC

![roc](roc.png)

### Loss VS Epoch / Accuracy VS Epoch

![epochs](epochs.png)

Podemos observar uma clara convergência por essa imagem ainda, platou após 15 epochs aproxidamente.

# 8. Avaliação do modelo

### Olhando os dados brutos

- Precision: 0.936
- Recall: 0.971
- Accuracy: **96.01%**

A matriz de confusão mostra ainda que a taxa de falsos positivos é maior que a taxa de falsos negativos, significando que é menos provável o modelo afirmar que um passageiro saiu satisfeito, quando estava insatisfeito, do que afirmar que ele saiu insatisfeito quando estava satisfeito, que é uma característica desejada.

Enquanto isso, a curva ROC mostra que a taxa de verdadeiros positivos é imensamente maior que a de falsos positivos, dando ainda maior validade ao modelo.

Por fim, devemos comparar nosso modelo atual com uma baseline para julgarmos a sua efetividade relativa:

### Regressão logística:

---

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)

print("Accuracy (LogReg):", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))
```

### Output
```
Accuracy (LogReg): 0.8717277486910995
AUC: 0.9269720121120558
```

Nosso modelo apresenta uma acurácia e um AUC (área de baixo do gráfico ROC, capacidade de separar as classes) significativamente maiores que a regressão logística (96% e 0.99 respectivamente). Demonstrando assim que a complexidade extra da rede neural é justificada, ainda, pelo fato de ambos modelos terem bons desempenhos, isso mostra que parte do sinal do nosso modelo é linear, já que a regressão linear segue em partes essa característica.

---

### Random forest:

---

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

print("Accuracy (RandomForest):", accuracy_score(y_test, preds))
```

### Output
```
Accuracy (RandomForest): 0.963658761934093
```

Nosso modelo apresenta uma acurácia análoga a do random forest, que é um modelo consolidade e validado já. Mostrando sua efetividade e mostrando como ele também é bom em capturar não linearidades, que é uma característica chave do random florest.

---

### Dummy (aleatório):

---

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
print("Accuracy (Dummy):", dummy.score(X_test, y_test))

```

### Output
```
Accuracy (Dummy): 0.5610178626424391
```

Nosso modelo tem uma acurácia significativamente maior do que a dummy, que é essencialmente favorecer a classe majoritária (que tem mais quantidade numérica). Mostrando assim que nosso modelo é independente da sorte e está acima do baseline nulo

---

# Função de plot auxiliar:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_unused_axes(fig, axes, used):
    """Remove eixos não usados (vazios) do figure."""
    for j in range(used, len(axes)):
        fig.delaxes(axes[j])

def plot_distributions(df, quantitative_cols, ordinal_cols, nominal_cols,
                       title_suffix="", force_ordinal_continuous=False,
                       delay_log_scale=True):
    """
    Plota distribuições. 
    *********************ATENÇÃO:*********************
    Para colunas de 'delay' aplicamos np.log1p(clip(lower=0))
    para PRESERVAR zeros e comprimir caudas (equivalente a log(1+x)). Log(0) não existe, não podemos ignorar a maioria dos dados.
    """
    sns.set_style("white")

    if quantitative_cols:
        n = len(quantitative_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
        axes = axes.flatten()
        for i, col in enumerate(quantitative_cols):
            ax = axes[i]
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if delay_log_scale and "delay" in col.lower():
                series_plot = np.log1p(series.clip(lower=0))
                sns.histplot(series_plot, kde=False, ax=ax, stat="percent", bins=40)
                ax.set_xlabel(f"{col} (log1p)")
            else:
                sns.histplot(series, kde=False, ax=ax, stat="percent", bins=40)
            ax.set_title(col)
            ax.set_ylabel("Percent")
            ax.tick_params(axis="x", rotation=45)
        remove_unused_axes(fig, axes, len(quantitative_cols))
        fig.suptitle(f"Distribuição das Variáveis Quantitativas {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

    # --- Ordinais ---
    if ordinal_cols:
        n = len(ordinal_cols)
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
        axes = axes.flatten()
        for i, col in enumerate(ordinal_cols):
            ax = axes[i]
            coerced = pd.to_numeric(df[col], errors="coerce").dropna()
            if force_ordinal_continuous:
                sns.histplot(coerced, kde=False, ax=ax, stat="percent", bins=40)
            else:
                counts = coerced.value_counts(normalize=True).sort_index() * 100
                sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set_title(col)
            ax.set_ylabel("Percent")
            ax.tick_params(axis="x", rotation=45)
        remove_unused_axes(fig, axes, len(ordinal_cols))
        fig.suptitle(f"Distribuição das Variáveis Ordinais {title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(); plt.show()

    # --- Nominais ---
    if nominal_cols:
        # se lista vazia ou None, pula todo o bloco (evita n_rows=0)
        n = len(nominal_cols)
        if n > 0:
            n_cols = 3
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, max(1, n_rows)*4))
            axes = axes.flatten()
            used = 0
            for i, col in enumerate(nominal_cols):
                ax = axes[i]
                counts = df[col].astype(str).value_counts(dropna=False)
                labels = list(counts.index)
                sns.barplot(x=labels, y=(counts/counts.sum()*100).values, ax=ax)
                ax.set_title(col)
                ax.set_ylabel("Percent")
                ax.tick_params(axis="x", rotation=45)
                used += 1
            remove_unused_axes(fig, axes, used)
            fig.suptitle(f"Distribuição das Variáveis Nominais {title_suffix}", fontsize=16, y=1.02)
            plt.tight_layout(); plt.show()
```