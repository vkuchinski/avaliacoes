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

    dpt_min = np.float64(dependente.min())

    dpt_max = np.float64(dependente.max())

    ## gráfico de aderência
    print('Gráfico de aderência\n')

    fit_m, fit_b = np.polyfit(dependente_flat, previsao, 1)
    plt.plot(dependente, fit_m*dependente + fit_b, color = 'darkorange', linewidth=1, label = 'média')
    plt.plot([dpt_min,dpt_max],[dpt_min,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#000000', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.show()

    input('Pressione ENTER para continuar a análise...')

    ## grafico de dispersão dos residuos
    print('\nGráfico de dispersão dos resíduos\n')

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
    print('        Resultado Hipótese nº 6')

    for i in range(len(regressores.columns)):
        if abs(corr).iloc[i].ge(0.8).value_counts()[1] > 1:
            print('Não é atendida pela análise da correlação, verificar indícios de multicolinearidade na variável', corr.columns[i])
        else:
            print('É atendida pela análise da correlação, não há índicios de multicolinearidade na variável', corr.columns[i])

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