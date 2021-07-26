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
    
    dpt_min = np.float64(dependente.min())

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
    plt.plot([dpt_min,dpt_max],[dpt_min,dpt_max], color = 'black', linewidth=1, linestyle='--', label = 'bissetriz')
    plt.scatter(dependente, previsao, c='#000000', marker='o', s=5)
    plt.title('Gráfico de aderência entre os valores observados (x) e os valores previstos (y)')
    plt.savefig('.\output\saderencia.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # grafico de dispersão dos residuos
    plt.axhline(y=-2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='#45d4ff', linestyle='--', linewidth=1)
    plt.axhline(y=2, color='#ffa1a1', linestyle='--', linewidth=1)
    plt.scatter(previsao, res_pad, c='#000000',  marker='o', s=5)
    plt.title('Gráfico de dispersão entre os valor calculados (x) e os resíduos padronizados (y)')
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

            ic_calculado = pd.DataFrame({'IC(-) 80%':[np.float64(perc_icB*-100)],
                            'Tend. Central': [100],
                            'IC(+) 80%': [np.float64(perc_icA*100)]})

            espaco = pd.DataFrame({'IC(-) 80%':[''],
                            'Tend. Central': [''],
                            'IC(+) 80%': ['']})

            ca_titulo = pd.DataFrame({'IC(-) 80%':['Campo Arbitrio'],
                            'Tend. Central': ['Tend. Central'],
                            'IC(+) 80%': ['Campo Arbitrio']})

            ca_valores = pd.DataFrame({'IC(-) 80%':[np.float64(tend_central*0.85)],
                        'Tend. Central': [np.float64(tend_central)],
                        'IC(+) 80%': [np.float64(tend_central*1.15)]})

            ca_var = pd.DataFrame({'IC(-) 80%':['-15'],
                    'Tend. Central': [np.float64(100)],
                    'IC(+) 80%': ['+15']})

            tabela_resultado = tabela_resultado.append([ic_calculado], ignore_index=True)
            tabela_resultado = tabela_resultado.rename(index={0:'Calculado', 1:'Variação', 2:'-', 3:'Calculado', 4:'Variação'})
            tabela_resultado = tabela_resultado.append([espaco], ignore_index=True)
            tabela_resultado = tabela_resultado.append([ca_titulo], ignore_index=True)
            tabela_resultado = tabela_resultado.append([ca_valores], ignore_index=True)
            tabela_resultado = tabela_resultado.append([ca_var], ignore_index=True)
            tabela_resultado.to_csv('.\output\projecao.csv', index=False)
            ans_e = None

        elif ans_e=='2':
            print('\n  Não, apenas o relatório estatístico\n')  
            ans_e = None
        else:
            print('\n  Não é uma opção válida.\n  Tente novamente\n')

    print('\nOs arquivos foram salvos no seguinte diretório:\n', caminho)

    input('\nPressione ENTER para retornar...')