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

    dpt_min = np.float64(dependente.min())

    dpt_max = np.float64(dependente.max())

    ## gráfico de aderência
    print('Gráfico de aderência\n')

    fit_m, fit_b = np.polyfit(dependente_flat, previsao, 1)
    plt.plot(dependente, fit_m*dependente + fit_b, color = 'darkorange', linewidth=1, label = 'média')
    plt.plot([dpt_min,dpt_max],[dpt_min,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#45d4ff', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.show()

    input('Pressione ENTER para continuar a análise...')

    ## grafico de dispersão dos residuos
    print('\nGráfico de dispersão dos resíduos\n')

    plt.axhline(y=-2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='#45d4ff', linestyle='--', linewidth=1)
    plt.axhline(y=2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.scatter(previsao, res_pad, c='#000000',  marker='o', s=5)
    plt.title('Gráfico de dispersão entre os valores calculados (x) e os resíduos padronizados (y)')
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