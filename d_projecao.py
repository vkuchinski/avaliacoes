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
    print(tabela_resultado)

    input('\nPressione ENTER para retornar...')