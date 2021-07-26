####################################################################################
#                                                                                  #
#   Esse código foi desenvolvido no âmbito do Trabalho de Conclusão de Curso       #
#   do curso de Especialização em Auditoria, Avaliações e Perícias de Engenharia   #
#   do Instituto de Pós-Graduação - IPOG, turma AEPOA010                           #
#   Autor: Vinícius Kuchinski                                                      #
#   Versão do código: 1.0                                                          #
#   Última edição: 12/07/2021                                                      # 
#                                                                                  # 
####################################################################################

############################## etapa de pré-processamento
# importando as bibliotecas que serão utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from b_analise_saneamento import analise_saneamento as analise
from c_hipoteses import teste_hipoteses as hipoteses
from d_projecao import projecao as projetar
from e_salvando import exportando as export

# menu
ans=True
while ans:
    print('''
    1.Começar um novo modelo
    2.Realizar análise das estastísticas básicas e saneamento
    3.Checar as hipóteses básicas da regressão linear
    4.Realizar avaliação - projeção
    5.Exportar resultados
    6.Sair
    ''')
    ans=input('PAVIMUR - Programa de avaliações de imóveis urbanos\nO que você gostaria de fazer? Digite o número da opção\n    ')
    if ans=='1':
      print('\nComeçar um novo modelo\n')
      # importando os dados
      # data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

      print('Por favor, indique o caminho ou o nome do arquivo csv com a sua amostra')
      tabela = input()
      data = pd.read_csv(tabela, encoding = 'ISO-8859-1')

      # visualizando os dados
      data.describe()
      print('')
      print(data.head())

      # selecionando as variáveis
      input_indep = input('\nDigite as variáveis independentes escolhidas, separadas por espaço: ')
      selec_indep = input_indep.split(" ")
      print('')

      input_dep = input('Digite a variável dependente escolhida: ')
      selec_dep = input_dep.split(" ")
      print('')

      X = data[selec_indep]
      y = data[selec_dep]

      # salvando as variáveis em uma nova tabela
      dados_s = pd.concat([X, y], axis=1)

      # visualizando as variáveis independentes versus a variável dependente para verificar necessidade de transformar variáveis
      print('Gráficos entre variáveis independentes vs. variável dependente para verificar a necessidade de transformação das variáveis antes do ajuste do modelo\n')

      input('Pressione ENTER para imprimir...')

      fig, ax = plt.subplots(1, len(X.columns), figsize = (10, 2))

      for idx, col in enumerate(X, 0):
          ax[idx].plot(X[col], y, 'o', markersize=2)
          ax[idx].set_title(col)
      plt.show()

      # ajustando o modelo
      print('\nDigite a equação de ajuste, conforme a estrutura de exemplo a seguir:')
      print('variavel dependente ~ np.log(var1) + np.reciprocal(var2) + var3')
      print('        |           |         |                |             |')
      print('    unitario        |     transfor. ln(x)      |       s/ tranform.')
      print('             símbolo ajuste              tranfor. 1/x')
      print('Ex:')
      print('unitario ~ np.log(area) + np.reciprocal(frente) + renda_bairro + ... ')
      print('')
      print('Variáveis escolhidas:',selec_indep, 'e', selec_dep, '\n')
      formula = input('Digite a equação:\n    ')

      equacao = smf.ols(formula, data=dados_s).fit()

      print('\nEquação armazenada\n')

    elif ans=='2':
      print('\nRealizar análise das estastísticas básicas e saneamento\n')
      dados_s = dados_s.dropna()
      dados_s = dados_s.reset_index(drop=True)
      X = dados_s[selec_indep]
      y = dados_s[selec_dep]
      equacao = smf.ols(formula, data=dados_s).fit()
      dados_s = analise(equacao, X, y, dados_s)
   
    elif ans=='3':
      print('\nChecar as hipóteses básicas da regressão linear\n')
      dados_s = dados_s.dropna()
      dados_s = dados_s.reset_index(drop=True)
      X = dados_s[selec_indep]
      y = dados_s[selec_dep]
      equacao = smf.ols(formula, data=dados_s).fit()      
      hipoteses(equacao, X, y, dados_s)

    elif ans=='4':
      print('\nRealizar avaliação - projeção\n') 
      dados_s = dados_s.dropna()
      dados_s = dados_s.reset_index(drop=True)
      X = dados_s[selec_indep]
      y = dados_s[selec_dep]
      equacao = smf.ols(formula, data=dados_s).fit()
      projetar(equacao, X)

    elif ans=='5':
          print('\n Exportar resultados\n')
          export(equacao, X, y, dados_s)

    elif ans=='6':
      print('\n Saindo...\n') 
      ans = None

    else:
      print('\nNão é uma opção válida.\nTente novamente')