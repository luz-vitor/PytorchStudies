# Classificação de Tipos de Flor Iris com PyTorch

Este é um projeto de classificação de tipos de flor Iris usando a biblioteca PyTorch em Python. O objetivo deste projeto é treinar uma rede neural para classificar flores Iris em três categorias diferentes: Setosa, Versicolor e Virginica.

## Dataset

O conjunto de dados utilizado para este projeto foi obtido no site [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). O conjunto de dados contém medidas de comprimento e largura da sépala e da pétala de 150 flores Iris, sendo 50 de cada tipo.

## Guia de Referência

O desenvolvimento deste projeto foi guiado por uma série de tutoriais no YouTube. Você pode encontrar o tutorial em [YouTube - Deep Learning With PyTorch](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1). O tutorial fornece instruções básicas sobre como criar uma rede neural utilizando PyTorch e treiná-la para classificar flores Iris.

## Estrutura do Projeto

O projeto está estruturado da seguinte forma:

- `model.py`: Contém a definição da classe `Model`, que é a rede neural utilizada no projeto. A rede possui camadas de entrada, duas camadas ocultas e uma camada de saída para classificar as flores Iris em três categorias diferentes (Setosa, Versicolor e Virginica). A arquitetura da rede pode ser personalizada ajustando os parâmetros no construtor da classe.

- `train.py`: O código Python responsável por treinar a rede neural. Ele carrega o conjunto de dados Iris, converte os dados para tensores PyTorch, define a função de perda (criterion) e o otimizador (optimizer) para o treinamento. O treinamento é realizado por um número especificado de épocas (epochs), com o acompanhamento do erro ao longo do tempo.

- `test.py`: Este script avalia o desempenho da rede neural treinada no conjunto de testes. Ele carrega o modelo treinado a partir do arquivo "my_iris_model.pt", faz previsões no conjunto de testes e compara as previsões com os rótulos reais para calcular a precisão da rede.

- `iris_dataset.csv`: O arquivo CSV contendo os dados do conjunto de dados Iris. Este arquivo deve ser baixado do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) e colocado na raiz do projeto.

- `requirements.txt`: Um arquivo contendo as dependências necessárias para executar o código. Certifique-se de instalá-las executando `pip install -r requirements.txt`.

- `my_iris_model.pt`: O arquivo que contém os pesos treinados da rede neural após o treinamento. É usado para carregar o modelo treinado em `test.py` e fazer previsões.

- `README.md`: Este arquivo de documentação que fornece informações sobre o projeto.


## Requisitos de Instalação

Certifique-se de ter as seguintes bibliotecas instaladas no seu ambiente Python:

```bash
pip install -r requirements.txt
````
## Ferramentas e Recursos Utilizados

- [Python](https://www.python.org/)

- [PyTorch](https://pytorch.org/)

- [Jupyter Notebook](https://jupyter.org/)

- [Matplotlib](https://matplotlib.org/)

- [Pandas](https://pandas.pydata.org/)

- [scikit-learn](https://scikit-learn.org/)

- [YouTube Tutorial](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1)

- [Readme.so](https://readme.so/pt)
