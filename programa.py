####################################################################################
#                                                                                  #
#   Esse código foi desenvolvido no âmbito do Trabalho de Conclusão de Curso       #
#   do curso de Especialização em Auditoria, Avaliações e Perícias de Engenharia   #
#   do Instituto de Pós-Graduação - IPOG, turma AEPOA010                           #
#   Autor: Vinícius Kuchinski                                                      #
#   Versão do código: 0.5                                                          #
#   Última edição: 05/julho/2021                                                      # 
#                                                                                  # 
####################################################################################

############################## etapa de pré-processamento
# importando as bibliotecas que serão utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

def analise_saneamento(regressao, regressores, dependente, dados):
    ############################## análises do modelo escolhido

    # importando as bibliotecas que serão utilizadas
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    ############################## análise das estatísticas básicas
    # inspecionando o sumário estatístico 
    print(regressao.summary(alpha = 0.20))
    print('')
    ## transformando-os em tabela para facilitar a manipulação
    sumario0 = (regressao.summary2().tables[0])
    sumario1 = (regressao.summary2().tables[1])
    sumario2 = (regressao.summary2().tables[2])
    type(sumario0)
    type(sumario1)
    type(sumario2)

    # retirando as informações estatísticas importantes para análise

    input('Pressione ENTER para continuar a análise...\n')

    r2 = sumario0.iloc[6][1]
    print('O valor do R2 é:', r2)
    print('')
    ar2 = sumario0.iloc[0][3]
    print('O valor do R2 Ajustado é:', ar2)
    print('')
    Fcalc = sumario0.iloc[4][3]
    print('O valor do F calculado é:', Fcalc)
    print('')
    sigF = sumario0.iloc[5][3]
    print('O valor da significância global é:', sigF)
    print('')

    input('Pressione ENTER para continuar a análise...\n')

    print('Significância bicaudal dos regressores, em %:')
    for i in range(len(regressores.columns)):
        if sumario1.iloc[i+1][3]*100 >= 30:
            print('    {:.2f}'.format(sumario1.iloc[i+1][3]*100), 'para a variável', sumario1.index[i+1], '- Atenção! Acima do permitido para Grau I')

        else:
            print('    {:.2f}'.format(sumario1.iloc[i+1][3]*100), 'para a variável', sumario1.index[i+1])

    input('\nPressione ENTER para continuar a análise...\n')

    print('A equação do modelo é: \ny = {:.5}'.format(sumario1.iloc[0][0]), '+')
    for i in range(len(regressores.columns)):
        print('{:.5} *'.format(dados.columns.values[i]), '( {:.5}'.format(sumario1.iloc[i+1][0]), ') +')

    input('\nPressione ENTER para continuar a análise...\n')

    ## gráficos básicos
    previsoes = regressao.get_prediction(dados, transform=True)

    previsao = pd.DataFrame(previsoes.summary_frame(alpha=0.20))

    previsao = pd.DataFrame(previsao['mean'])

    previsao.columns.values[0]=dependente.columns[0]

    residuos =  dependente.reset_index(drop=True) - previsao.reset_index(drop=True)

    residuos.dropna()

    desvpad = np.std(residuos)

    res_pad = residuos/desvpad

    res_pad.dropna()

    dependente_flat = dependente.values.flatten()

    dpt_max = np.float64(dependente.max())

    ## gráficos de aderência
    print('Gráfico de aderência\n')

    fit_m, fit_b = np.polyfit(dependente_flat, previsao, 1)
    plt.plot(dependente, fit_m*dependente + fit_b, color = 'darkorange', linewidth=1, label = 'média')
    plt.plot([0,dpt_max],[0,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#45d4ff', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.show()

    input('Pressione ENTER para continuar a análise...')

    print('\nGráfico de dispersão dos resíduos\n')

    ## grafico de dispersão dos residuos
    plt.axhline(y=-2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='#45d4ff', linestyle='--', linewidth=1)
    plt.axhline(y=2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.scatter(previsao, res_pad, c='#000000',  marker='o', s=5)
    plt.title('Gráfico de dispersão entre os unitários (x) e os resíduos padronizados (y)')
    plt.show()

    input('Pressione ENTER para continuar a análise...')

    ans_b=True
    while ans_b:
        print('''
    Gostaria de excluir outliers baseado em um limiar?
        1.Sim, excluir outliers
        2.Não, retornar ao menu principal
        ''')
        ans_b=input('Responda conforme as opções acima: \n    ')
        if ans_b=='1':
            print('\n  Sim, excluir outliers\n')
            dados_mod = dados.assign(ResPad=res_pad)

            dados_mod_row = dados_mod.shape[0]

            limiar = np.float64(input('Indique o limiar em desvios padrães para a remoção dos dados fora desse intervalo\n    '))

            dados_filtrados = dados_mod.drop(dados_mod[(dados_mod.ResPad < -(limiar)) | (dados_mod.ResPad > limiar)].index)
            dados_filtrados.describe()
            dados_filtrados_row = dados_filtrados.shape[0]
            
            print('')

            # percentual de outliers
            percent_outliers = ((dados_mod_row-dados_filtrados_row)/dados_mod_row)*100
            if percent_outliers < 5:
                print('Ok, o número de outliers está dentro dos 5% permitidos por conta da distribuição normal')
                input('\nPressione ENTER para encerrar...')

            else:
                print('Cuidado: O número de outliers excede os 5% permitidos por conta da distribuição normal')
                input('\nPressione ENTER para encerrar...')
            
            dados_fim = dados_filtrados.drop(['ResPad'], axis=1)
            dados_fim = dados_fim.reset_index(drop=True)

            return dados_fim

        elif ans_b=='2':
            print('\n  Não, retornar ao menu principal')  

            return dados
  
        else:
            print('\n  Não é uma opção válida.\n  Tente novamente\n')

def teste_hipoteses(regressao, regressores, dependente, dados):
    
    # importando as bibliotecas que serão utilizadas
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import seaborn as sns
    import statsmodels.formula.api as smf
    import statsmodels.stats.api as sms
    from statsmodels.compat import lzip
    from statsmodels.stats.diagnostic import normal_ad
    from statsmodels.stats.stattools import durbin_watson

    ############################## hipóteses básicas
    print('Verificação das hipóteses básicas da regressão linear múltipla, conforme Dantas (2012):\n')
    print('Hipótese 1 - Variável independente deve ser representada por números reais que não contenham nenhuma perturbação aleatória.')
    print('Hipótese 2 - O número de observações deve ser superior ao número de parâmetros estimados')
    print('Hipótese 3 - Os erros são variáveis aleatórias, com valor esperado nulo e variância constante')
    print('Hipótese 4 - Os erros são variáveis aleatórias, com distribuição normal')
    print('Hipótese 5 - Os erros não são correlacionados, isto é, são independentes sob a condição de normalidade')
    print('Hipótese 6 - Não deve existir nenhuma relação exata entre quaisquer variáveis independentes')
    print('')
    print('A seguir, as hipóteses serão verificadas, uma a uma.\n')
    
    input('Pressione ENTER para iniciar as análises...')

    # hipóteses básicas 1 - variável independente deve ser representada por números reais que não contém nenhuma perturbação aleatória
    print('')
    print('        Hipótese nº 1\nVariável independente deve ser representada por números reais que não contenham nenhuma perturbação aleatória.\n')

    print('        Resultado Hipótese nº 1\nÉ atendida, segundo Dantas (2012), pois dados imobiliários são números reais que não contém nenhuma perturbação aleatória.')

    print('')

    input('Pressione ENTER para continuar a análise...')

    # hipóteses básicas 2 - o número de observações, m, deve ser superior ao número de parâmetros estimados, isto é, para o caso de regressão linear simples deve ser superior a dois;
    print('')
    print('        Hipótese  nº 2\nO número de observações deve ser superior ao número de parâmetros estimados\n')

    if regressores.shape[0] < (3*regressores.shape[1]):
        print('        Resultado Hipótese nº 2\nNão é atendida, sua amostra é muito pequena e não atende critérios da hipotese (Dantas, 2012), bem como o item A.2, subitem a) da NBR 14653-2')
    else:
        print('        Resultado Hipótese nº 2\nÉ atendida, conforme os critérios de Dantas (2012), bem como do item A.2, subitem a) da NBR 14653-2')
    
    print('')

    input('Pressione ENTER para continuar a análise...')

    # hipóteses básicas 3 - os erros são variáveis aleatórias, com valor esperado nulo e variância constante
    print('')
    print('        Hipótese  nº 3\nOs erros são variáveis aleatórias, com valor esperado nulo e variância constante\n')

    ## calculando os resíduos, média e desvio padrão
    previsoes = regressao.get_prediction(dados, transform=True)

    previsao = pd.DataFrame(previsoes.summary_frame(alpha=0.20))

    previsao = pd.DataFrame(previsao['mean'])

    previsao.columns.values[0]=dependente.columns[0]

    residuos =  dependente.reset_index(drop=True) - previsao.reset_index(drop=True)

    residuos.dropna()

    media = np.mean(residuos)

    desvpad = np.std(residuos)

    res_pad = residuos/desvpad
    res_pad.dropna()

    dependente_flat = dependente.values.flatten()
    dependente_flat
    dpt_max = np.float64(dependente.max())

    ## gráficos de dispersão
    fit_m, fit_b = np.polyfit(dependente_flat, previsao, 1)
    plt.plot(dependente, fit_m*dependente + fit_b, color = 'darkorange', linewidth=1, label = 'média')
    plt.plot([0,dpt_max],[0,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#45d4ff', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.show()

    ## grafico de dispersão dos residuos
    plt.axhline(y=-2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='#45d4ff', linestyle='--', linewidth=1)
    plt.axhline(y=2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.scatter(previsao, res_pad, c='#000000',  marker='o', s=5)
    plt.title('Gráfico de dispersão entre os unitários (x) e os resíduos padronizados (y)')
    plt.show()

    GQ_name = ['F statistic', 'p-value']
    GQ_test = sms.het_goldfeldquandt(regressao.resid, regressao.model.exog)
    GQ_result = lzip(GQ_name, GQ_test)

    print('Faça uma análise gráfica para verificar homocedasticidade (comportamento aleatório) ou heterocedasticidade (agrupamento indesejado)')
    print('')

    if GQ_result[1][1] < 0.05:
        print('        Resultado Hipótese nº 3\nÉ atendida pelo teste de Goldfeld-Quandt, ou seja, há indícios de homocedasticidade')
    else:
        print('        Resultado Hipótese nº 3\nNão atendida pelo teste de Goldfeld-Quandt, ou seja, há indícios de heterocedasticidade')

    print('')
    
    input('Pressione ENTER para continuar a análise...')

    # hipóteses básicas 4 - os erros são variáveis aleatórias, com distribuição normal
    print('')
    print('        Hipótese  nº 4\nOs erros são variáveis aleatórias, com distribuição normal\n')

    ## gráfico do histograma
    plt.hist(residuos, bins=25, density=True, alpha=0.5, color='g')

    ## curva normal relacionada
    xmin, xmax = plt.xlim()
    curva_x = np.linspace(xmin, xmax, 100)
    curva_p = norm.pdf(curva_x, media, desvpad)
    plt.plot(curva_x, curva_p, 'k', linewidth=2)
    title = "Fit results: media = %.2f,  desvpad = %.2f" % (media, desvpad)
    plt.title(title)
    plt.show()
    print('Faça uma análise gráfica: um histograma de resíduos apresentando formato parecido com o da curva normal é um indicador favorável de hipótese de normalidade do erro')
    print('')

    ## análise através do teste estatístico de Anderson-Darling
    AD_test = normal_ad(residuos)[1]

    if AD_test < 0.05:
        print('        Resultado Hipótese nº 4\nNão é atendida pelo teste de Anderson-Darling, ou seja, índicios de não normalidade nos resíduos')
    else:
        print('        Resultado Hipótese nº 4\nÉ atendida pelo teste de Anderson-Darling, ou seja, índicios de normalidade nos resíduos')
    
    print('')

    input('Pressione ENTER para continuar a análise...')

    # hipóteses básicas 5 - os erros não são correlacionados, isto é, são independentes sob a condição de normalidade
    print('')
    print('        Hipótese  nº 5\nOs erros não são correlacionados, isto é, são independentes sob a condição de normalidade')

    ## teste Durbin-Watson
    dw = durbin_watson(regressao.resid)
    print('')
    print('O valor da estatística Durbin-Watson é:', dw)
    print('')
    print('Valores entre 1,5 e 2,5 podem ser considerados normais, ou sejam, sem autocorrelação entre os resíduos')
    print('Valores fora desse intervalo podem indicar autocorrelação, positiva se maior que 2,5 e negativa se menor que 1,5')
    print('')
    if dw < 1.5:
        print('Sinais de autocorrelação positiva')
        print('Se houver série temporal (utilizar data como variável), considere usar alguma variável de retardo para ajuste dos dados.')
        print('')
        print('        Resultado Hipótese nº 5\nNão é atendida pelo teste de Durbin-Watson')
    elif dw > 2.5:
        print('Sinais de autocorrelação negativa')
        print('Se houver série temporal (utilizar data como variável), considere usar alguma variável de retardo para ajuste dos dados.')
        print('')
        print('        Resultado Hipótese nº 5\nNão é atendida pelo teste de Durbin-Watson')
    else:
        print('De pequena a não autocorrelação')
        print('')
        print('        Resultado Hipótese nº 5\nÉ atendida pelo teste de Durbin-Watson')

    print('')
    
    input('Pressione ENTER para continuar a análise...')

    # hipóteses básicas 6 - não deve existir nenhuma relação exata entre quaisquer variáveis independentes
    print('')
    print('        Hipótese  nº 6\nNão deve existir nenhuma relação exata entre quaisquer variáveis independentes')

    ## matriz de correlação
    corr = dados.corr()

    ### criando o mapa de calor da matriz de correlação
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette('coolwarm', as_cmap=True)

    ### plotando o mapa
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": 0.5})

    heatmap.set_title('Correlação entre variáveis', fontdict={'fontsize':12}, pad=12)
    plt.show()

    ## mensagem de alerta
    print('Atenção para valores de correlação maiores do que 0,80')

    print('')

    for i in range(len(regressores.columns)):
        if abs(corr).iloc[i].ge(0.8).value_counts()[1] > 1:
            print('        Resultado Hipótese nº 6\nNão é atendida pela análise da correlação, verificar indícios de multicolinearidade na variável', corr.columns[i])
        else:
            print('        Resultado Hipótese nº 6\nÉ atendida pela análise da correlação, não há índicios de multicolinearidade na variável', corr.columns[i])

        ## calculo_do_vif 
    ### criando os dicionários
    vif_dict, tolerance_dict = {}, {}

    ### criando a fórmula para cada uma das variáveis exógenas
    for exog in regressores:
        not_exog = [i for i in regressores if i != exog]
        eqc = f"{exog} ~ {' + '.join(not_exog)}"

        # extraíndo o r2
        r_squared = smf.ols(eqc, data=dados).fit().rsquared

        # calculando o VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculando a tolerância
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    ### inserindo os valores em um dataframe
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    ### apresentando os resultados do VIF
    print('')
    print('Segundo o teste Fator de Inflação da Variância, tem-se:')
    for i in range(len(regressores.columns)):
        if df_vif.iloc[i][0] > 10:
            print('{:.3}'.format(df_vif.iloc[i][0]), 'para a variável', df_vif.index[i], '- há a possibilidade de multicolinearidade (VIF > 10)')
        elif df_vif.iloc[i][0] > 100:
            print('{:.3}'.format(df_vif.iloc[i][0]), 'para a variável', df_vif.index[i], '- com certeza existe multicolinearidade (VIF > 100)')
        else:
            print('{:.3}'.format(df_vif.iloc[i][0]), 'para a variável', df_vif.index[i], '- não há indício de multicolinearidade (VIF < 10)')

def projecao(regressao, regressores):
    
    # importando as bibliotecas que serão utilizadas
    import pandas as pd
    import numpy as np
    
    ############################## projeção
    projecao = pd.DataFrame(columns = regressores.columns)
    list_proj = []

    for i in range(len(regressores.columns)):
        print("Entre com o valor da variável", regressores.columns[i])
        item = np.float64(input('    '))
        list_proj.append(item)
        
    projecao.loc[0] = list_proj

    res_proj = regressao.get_prediction(projecao).summary_frame(alpha = 0.20)

    tend_central = res_proj['mean']
    res_proj_ic_abaixo = res_proj['mean_ci_lower']
    res_proj_ic_acima = res_proj['mean_ci_upper']
    perc_icB = (res_proj['mean']-res_proj['mean_ci_lower'])/res_proj['mean']
    perc_icA = (res_proj['mean_ci_upper']-res_proj['mean'])/res_proj['mean']

    tabela_resultado = pd.DataFrame(np.column_stack([res_proj_ic_abaixo, tend_central, res_proj_ic_acima]))
    tabela_resultado.columns = ['IC(-) 80%', 'Tend. Central', 'IC(+) 80%']

    ic_calculado = pd.DataFrame({'IC(-) 80%':[np.float64(perc_icB)],
                    'Tend. Central': [1],
                    'IC(+) 80%': [np.float64(perc_icA)]})

    tabela_resultado = tabela_resultado.append([ic_calculado], ignore_index=True)
    tabela_resultado = tabela_resultado.rename(index={0:'Calculado', 1:'Variação'})
    print(tabela_resultado)

    input('\nPressione ENTER para retornar...')

def exportando(regressao, regressores, dependente, dados):
    ############################## análises do modelo escolhido

    # importando as bibliotecas que serão utilizadas
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import seaborn as sns
    import os

    #criando o diretório de saída
    pasta_nova = 'output'
    caminho_ant = os.getcwd()
    caminho = os.path.join(caminho_ant, pasta_nova) 
    os.makedirs(caminho, exist_ok = True)

    # sumario da regressao
    sumario0 = pd.DataFrame((regressao.summary2().tables[0]))
    sumario1 = pd.DataFrame((regressao.summary2().tables[1]))
    sumario2 = pd.DataFrame((regressao.summary2().tables[2]))

    sumario0.to_csv('.\output\sumario1.csv', index=False)
    sumario1.to_csv('.\output\sumario2.csv', index=True)
    sumario2.to_csv('.\output\sumario3.csv', index=False)

    dfeq = pd.DataFrame({'Intercepto': [sumario1.iloc[0][0]]})

    for i in range(len(regressores.columns)):
        dfeq.insert(i+1, str(regressores.columns[i]), sumario1.iloc[i+1][0])

    dfeq.to_csv('.\output\equacao.csv', index=False)

    dados_utilizados = pd.DataFrame(dados)

    dados_utilizados.to_csv('.\output\dados_utilizados.csv', index=False)

    #calculos para a execucao dos graficos
    previsoes = regressao.get_prediction(dados, transform=True)

    previsao = pd.DataFrame(previsoes.summary_frame(alpha=0.20))

    previsao = pd.DataFrame(previsao['mean'])

    previsao.columns.values[0]=dependente.columns[0]

    residuos =  dependente.reset_index(drop=True) - previsao.reset_index(drop=True)

    residuos.dropna()

    media = np.mean(residuos)

    desvpad = np.std(residuos)

    res_pad = residuos/desvpad
    res_pad.dropna()

    dependente_flat = dependente.values.flatten()
    dependente_flat
    dpt_max = np.float64(dependente.max())

    # matriz de correlação
    corr = dados.corr()

    # criando o mapa de calor da matriz de correlação
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.color_palette('coolwarm', as_cmap=True)

    # plotando o heatmap da correlacao
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": 0.5})

    heatmap.set_title('Correlação entre variáveis', fontdict={'fontsize':12}, pad=12)
    plt.savefig('.\output\correlacao.png', dpi=300, bbox_inches='tight')
    plt.clf()

    ## gráfico do histograma
    plt.hist(residuos, bins=25, density=True, alpha=0.5, color='g')

    # curva normal relacionada
    xmin, xmax = plt.xlim()
    curva_x = np.linspace(xmin, xmax, 100)
    curva_p = norm.pdf(curva_x, media, desvpad)
    plt.plot(curva_x, curva_p, 'k', linewidth=2)
    title = "Fit results: media = %.2f,  desvpad = %.2f" % (media, desvpad)
    plt.title(title)
    plt.savefig('.\output\histograma.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # gráficos de aderencia
    fit_m, fit_b = np.polyfit(dependente_flat, previsao, 1)
    plt.plot(dependente, fit_m*dependente + fit_b, color = 'darkorange', linewidth=1, label = 'média')
    plt.plot([0,dpt_max],[0,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#45d4ff', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.savefig('.\output\saderencia.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # grafico de dispersão dos residuos
    plt.axhline(y=-2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='#45d4ff', linestyle='--', linewidth=1)
    plt.axhline(y=2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.scatter(previsao, res_pad, c='#000000',  marker='o', s=5)
    plt.title('Gráfico de dispersão entre os unitários (x) e os resíduos padronizados (y)')
    plt.savefig('.\output\dispersao.png', dpi=300, bbox_inches='tight')
    plt.clf()

    #projecao

    ans_e=True
    while ans_e:
        print('''
    Gostaria de salvar resultados de uma projeção?
        1.Sim, salvar a tabela dos resultados de uma projeção
        2.Não, apenas o relatório estatístico
        ''')
        ans_e=input('Responda conforme as opções acima: \n    ')
        if ans_e=='1':
            print('\n  Sim, salvar a tabela dos resultados de uma projeção\n')

            projecao = pd.DataFrame(columns = regressores.columns)
            list_proj = []

            for i in range(len(regressores.columns)):
                print("Entre com o valor da variável", regressores.columns[i])
                item = np.float64(input('    '))
                list_proj.append(item)
                
            projecao.loc[0] = list_proj

            res_proj = regressao.get_prediction(projecao).summary_frame(alpha = 0.20)

            tend_central = res_proj['mean']
            res_proj_ic_abaixo = res_proj['mean_ci_lower']
            res_proj_ic_acima = res_proj['mean_ci_upper']
            perc_icB = (res_proj['mean']-res_proj['mean_ci_lower'])/res_proj['mean']
            perc_icA = (res_proj['mean_ci_upper']-res_proj['mean'])/res_proj['mean']

            tabela_resultado = pd.DataFrame(np.column_stack([res_proj_ic_abaixo, tend_central, res_proj_ic_acima]))
            tabela_resultado.columns = ['IC(-) 80%', 'Tend. Central', 'IC(+) 80%']

            ic_calculado = pd.DataFrame({'IC(-) 80%':[np.float64(perc_icB)],
                            'Tend. Central': [1],
                            'IC(+) 80%': [np.float64(perc_icA)]})

            tabela_resultado = tabela_resultado.append([ic_calculado], ignore_index=True)
            tabela_resultado = tabela_resultado.rename(index={0:'Calculado', 1:'Variação'})
            tabela_resultado.to_csv('.\output\projecao.csv', index=False)
            ans_e = None

        elif ans_e=='2':
            print('\n  Não, apenas o relatório estatístico\n')  
            ans_e = None
        else:
            print('\n  Não é uma opção válida.\n  Tente novamente\n')

    print('\nOs arquivos foram salvos no seguinte diretório:\n', caminho)

    input('\nPressione ENTER para retornar...')

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
    ans=input('Algoritmo de avaliações de imóveis urbanos\nO que você gostaria de fazer?\n    ')
    if ans=='1':
        print('\nComeçar um novo modelo\n')
        # importando os dados
        # data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

        tabela = input('Por favor, indique o caminho ou o nome do arquivo csv com a sua amostra\n    ')
        data = pd.read_csv(tabela)

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
            ax[idx].plot(y, X[col], 'o', markersize=2)
            ax[idx].set_xlabel(selec_dep)
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

    elif ans=='2':
        print('\nRealizar análise das estastísticas básicas e saneamento\n')
        dados_s = dados_s.dropna()
        dados_s = dados_s.reset_index(drop=True)
        X = dados_s[selec_indep]
        y = dados_s[selec_dep]
        equacao = smf.ols(formula, data=dados_s).fit()
        dados_s = analise_saneamento(equacao, X, y, dados_s)
   
    elif ans=='3':
        print('\nChecar as hipóteses básicas da regressão linear\n')
        dados_s = dados_s.dropna()
        dados_s = dados_s.reset_index(drop=True)
        X = dados_s[selec_indep]
        y = dados_s[selec_dep]
        equacao = smf.ols(formula, data=dados_s).fit()      
        teste_hipoteses(equacao, X, y, dados_s)

    elif ans=='4':
        print('\nRealizar avaliação - projeção\n') 
        dados_s = dados_s.dropna()
        dados_s = dados_s.reset_index(drop=True)
        X = dados_s[selec_indep]
        y = dados_s[selec_dep]
        equacao = smf.ols(formula, data=dados_s).fit()
        projecao(equacao, X)

    elif ans=='5':
        print('\n Exportar resultados\n')
        exportando(equacao, X, y, dados_s)

    elif ans=='6':
      print('\n Saindo...\n') 
      ans = None

    else:
      print('\nNão é uma opção válida.\nTente novamente')
