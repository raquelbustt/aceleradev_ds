#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# O data set <b>black_friday</b> é uma disponibilização da Analytics Vidhya acessível através do Kaggle. Segundo a plataforma da Codenation, o data set traz algumas variáveis relativas à transações comerciais realizadas durante a Black Friday em uma determinada loja de varejo. Cada observação é relativa a um determinado item comprado por um usuário. O objetivo deste documento não é explorar os dados do dataframe, mas sim responder as questões propostas pelo desafio 1 do <b>ACELERADEV Data Science</b>. Portanto, não houve qualquer alteração no documento, exceto a complementação com as respostas necessárias para cada questão.

# In[12]:


black_friday.head(3)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# Este exercício poderia ser resolvido de duas formas, a primeira utilizando apenas o atributo <i>shape</i> que retornaria a quantidade de linhas e a quantidade de colunas do dataframe. Contudo, crei duas variáveis para que possamos ver que é possível mostrar separadamente a quantidade de linhas, através do indicador [0] e a quantidade de colunas com o indicador [1]. Por fim, foi reservado no atributo <i>tupla</i> o valor de ambos.

# In[3]:


def q1():
    n_observacoes = black_friday.shape[0]
    n_colunas = black_friday.shape[1]
    tupla = (n_observacoes, n_colunas)
    return (tupla)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# Novamente usamos o atributo <i>shape</i> para retornar a contagem de linhas, mas dessa vez precisamos de um filtro. Para isso, é usada a forma abaixo, que faz um "corte" dentro do dataframe, criando uma lista com True e False para todas as linhas que atendem as condições descritas, ou seja, mulheres entre 26-35 anos.

# In[4]:


def q2():
    return black_friday[(black_friday['Gender']=='F') & (black_friday['Age']=='26-35')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# O método <i>nunique</i> provém da biblioteca do numpy (importada no início do código). Seu objetivo é contar quantos valores únicos há em um array. Usando como parâmetro a coluna "User_ID" é fácil descobrir quantos valores usuários únicos há.

# In[5]:


def q3():
    return int(black_friday["User_ID"].nunique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# Mais uma vez o método <i>unique</i> pode nos salvar. Dessa vez, é utilizado em conjunto com um recurso de outra biblioteca (pandas). O <i>dtypes</i> retorna quais são os tipos de dados existentes em um determinado conjunto de dados. Abaixo então vemos que, é selecionado os tipos (dtypes) únicos(nunique) do dataframe <i>black_friday</i> e para garantir que será um valor único escalar, utiliza-se o int no início.

# In[6]:


def q4():
    return int(black_friday.dtypes.nunique())   


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# O valor percentual de uma determinada ocorrência é calculado a partir de <b>x/total de casos</b>, onde x é o parâmetro utilizado (proposto pelo exercício). Vejamos, o <i>shape</i>, como já visto, faz uma contagem de linhas, enquanto que o <i>dropna</i> remove os valores faltantes. Podemos então utilizar a técnica da primeira questão, fazer um "corte" no nosso dataframe a partir deste filtro. Você deve estar se questionando "mas isso não trará então o percentual dos valores preenchidos?", por isso utiliza-se o <b>1-</b> no início da equação, ou seja, o valor complementar.

# In[7]:


def q5():
    percentual = 1-(black_friday.dropna().shape[0]/black_friday.shape[0])
    return percentual


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# <p align="justify">Na fórmula abaixo temos uma série de métodos para resolver este problema. Se, por curiosidade, fizer o teste executando cada um dos métodos, um de cada vez, é possível perceber o filtro que cada método faz. Inicialmente o <i>isnull</i> identifica quais são os dados são nulos de forma booleana, então <i>sum</i> mostra a somatória dos nulls em cada variável e o <i>max</i> finalmente retorna qual é coluna possui mais dados nulos. Lembrando que essa forma é para alcançar o resultado pedido no exercício, porém, esses métodos podem ser executados de forma separa para outras finalidades.</p>

# In[12]:


def q6():
    return black_friday.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# Este exercício pode ser realizado a partir do que foi aplicado na questão 2, com o método <i>dropna</i>, recorta-se os valores faltantes. O que foi feito abaixo então:
# <li>Atribuiu-se um novo dataframe <b>bf</b> para não perder o dataframe anterior;
# <li>Utilizou-se o método <i>mode</i> que retorna qual é o valor mais frequente;
# <li>Mudou-se este retorno para inteiro, afim de retornar um único valor escalar.</li>
# Assim então, tem-se que o valor mais frequente é: 16

# In[9]:


def q7():
    bf = black_friday.dropna()
    return int(bf['Product_Category_3'].mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# A normalização ajusta os valores que estão em diferentes medidas para uma escala fictícia comum, entre -1 e 1, onde resultados entre -1 e 0, representam que o valor puro era negativo.<br>
# Nas importações, é possível ver que além do pandas e numpy, também foi o <i>preprocessing</i> da biblioteca <i>sklearn</i>. Através dele é possível executar a normalização de forma mais prática.
# <li>Primeiro foi retornado como um array em <b>x</b> a coluna "Purchase"<li>Em seguida, o método <i>MinMaxScaler</i> transforma cada elemento, de modo que esteja no intervalo especificado no conjunto. Como não está estabelecido, por default ele considera entre 0 e 1.<li>Em <b>x_scaled</b> os dados são ajustados e transformados em um array
# <li>E por fim, é retornado a média deste array

# In[10]:


def q8():
    x = black_friday['Purchase'].values #retorna um array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.reshape(-1,1))
    return x_scaled.mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variável `Purchase` após sua padronização? Responda como um único escalar.

# A padronização consiste em subtrair de um valor de um conjunto de dados a sua média e dividir o resultado pelo desvio padrão do conjunto ou variável. Novamente, será utilizado recursos do <i>preprocessing</i>, mas agora quem irá nos auxiliar é o método <i>StandardScaler()</i> que faz a padronização de forma simples. Os passos são os seguintes:
# <li>Primeiro foi retornado como um array em <b>x</b> a coluna "Purchase" em y<li>Em seguida, o método <i>StandardScaler</i> transforma cada elemento<li>Em <b>y_scaled</b> os dados são ajustados e transformados em um array<li>Em <b>ocorrencias</b> é filtrado quais elementos estão entre -1 e 1<li>E então, é retornado o tamanho (pelo <i>len</i>) este valor único.

# In[1]:


def q9():
    from sklearn.preprocessing import StandardScaler
    y = black_friday['Purchase'].values
    standard = StandardScaler()
    y_standard = standard.fit_transform(y.reshape(-1, 1))
    ocorrencias = y_standard[(y_standard > -1) & (y_standard < 1)]
    return len (ocorrencias)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# Existem N formas para fazer essa comparação, uma delas é criando dois atributos auxiliares (<b>aux1</b> e <b>aux2</b>) que retorna valores booleanos, sendo True caso o elemento seja nulo e caso o contrário, o retorno é False. Depois basta comparar as linhas pelo <i>shape[0]</i> de aux1 e aux 2 para saber se são iguais, ou seja, se uma observação é null em `Product_Category_2` ela também é em `Product_Category_3`.

# In[10]:


def q10():
    aux1 = black_friday['Product_Category_2'].isnull()
    aux2 = black_friday['Product_Category_3'].isnull()
    return bool (aux1.shape[0] == aux2.shape[0])


# In[11]:


q10()

