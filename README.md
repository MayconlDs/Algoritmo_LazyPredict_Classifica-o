
# Explorando Algoritmos de Classificação com LazyPredict e PrettyTable

Este projeto demonstra como automatizar a comparação de vários algoritmos de classificação usando a biblioteca `LazyPredict` e exibir os resultados de forma organizada com `PrettyTable`. Este processo é extremamente útil para cientistas de dados que desejam avaliar rapidamente diferentes modelos e escolher o melhor para seu problema.

## Descrição do Projeto

1. **Carregar Dados**: Utilizamos o conjunto de dados `breast_cancer` do `sklearn.datasets`, que contém informações sobre características de células tumorais.
2. **Dividir os Dados**: Separamos os dados em conjuntos de treino e teste para avaliar a performance dos modelos.
3. **Treinar Modelos**: Usamos o `LazyClassifier` para treinar e avaliar automaticamente vários modelos de classificação.
4. **Exibir Resultados**: Utilizamos o `PrettyTable` para criar uma tabela organizada com as métricas de performance de cada modelo.

## Código

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from prettytable import PrettyTable

# Carregar dados e dividir em treino e teste
dados = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(dados.data, dados.target, test_size=0.5, random_state=42)

# Treinar modelos usando LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Criar tabela usando prettytable
table = PrettyTable()
table.field_names = ["Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"]

for model in models.index:
    row = models.loc[model].values.tolist()
    table.add_row([model] + row)

print(table)
```

## Dependências

As seguintes bibliotecas são necessárias para executar o código:

- scikit-learn
- lazypredict
- prettytable

## Como Executar

1. Clone o repositório:
    ```
    git clone https://github.com/seuusuario/seurepositorio.git
    ```

2. Instale as dependências:
    ```
    pip install -r requirements.txt
    ```

3. Execute o script Python para ver os resultados:
    ```
    python seu_script.py
    ```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a Licença MIT.
