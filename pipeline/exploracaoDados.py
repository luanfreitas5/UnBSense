#!/usr/bin/env python
# coding: utf-8
"""

Author: Luan Freitas
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilitarios.utilitarios import getDiretorioPlots 
from utilitarios.colorPalettes import colorblind


class ExploracaoDados():
    """_summary_
    """

    def __init__(self):
        """Constructor
        """
        pass
        
    def graficoBarras(self, periodos=['pre_pandemia', 'pandemia']):
        """_summary_

        Args:
            periodos (list, optional): _description_. Defaults to ['pre_pandemia', 'pandemia'].
        """
        for periodo in periodos:
        
            diretorio = getDiretorioPlots()
            
            df = pd.read_csv(f'datasets/twitterbase_{periodo}.csv', sep=';')
            x = df['classe'].value_counts()
            ax = sns.barplot(x.index, x, palette=colorblind)
            
            
            ax.bar_label(ax.containers[0])
            ax.set_title(f'Distribuição entre as Classes Depressão e Controle no Periodo {periodo}')
            ax.set_xlabel('Classe')
            ax.set_ylabel('Quantidade')
            plt.savefig(diretorio + f'{periodo}_graficoBarrasClasses')
            # plt.show()
            plt.close()
        
    def graficoPizza(self, periodos=['pre_pandemia', 'pandemia']):
        """_summary_

        Args:
            periodos (list, optional): _description_. Defaults to ['pre_pandemia', 'pandemia'].
        """
        for periodo in periodos:
        
            diretorio = getDiretorioPlots()
            
            df = pd.read_csv(f'datasets/twitterbase_{periodo}.csv', sep=';')
            df = df['classe'].value_counts()
            #df.plot.pie(autopct="%.2f%%", title=f'Balanceamento entre as Classes Depressão e Controle no Periodo {periodo}')
            plt.pie(df, labels = ['depressão', 'controle'], autopct="%.2f%%", colors=colorblind)
            plt.title(f'Balanceamento entre as Classes Depressão e Controle no Periodo {periodo}')
            plt.savefig(diretorio + f'{periodo}_graficoPizzaClasses')
            # plt.show()
            plt.close()
