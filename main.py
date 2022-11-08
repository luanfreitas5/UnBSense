#!/usr/bin/env python
# coding: utf-8
"""
Modulo Principal
Author: Luan Freitas
"""
from metodologia.metodologia import Metodologia
from datetime import datetime
from utilitarios.utilitarios import strfdelta, carregarParametros
from traceback import print_exc



if __name__ == '__main__':
        
    try:
        start_time = datetime.now()
        metodologia = Metodologia()
        listaPeriodos, listaGrupos, mode = carregarParametros()

        if mode == 'buscar':
            metodologia.etapa_1_buscar_usuarios(listaPeriodos, listaGrupos)
        elif mode == 'coletar':
            metodologia.etapa_2_coletar_tweets(listaPeriodos, listaGrupos)
        elif mode == 'mesclar':
            metodologia.etapa_3_1_mesclar_arquivos(listaPeriodos, listaGrupos)
        elif mode == 'limpar':
            metodologia.etapa_3_2_limpar_arquivos(listaPeriodos, listaGrupos)
        elif mode == "atributos":
            metodologia.etapa_4_extrair_caracteristicas(listaPeriodos, listaGrupos)
        elif mode == "qualidade":
            metodologia.etapa_5_qualidade_Dados(listaPeriodos, listaGrupos)
        elif mode == 'vetores': 
            metodologia.etapa_6_vetores_caracteristicas(listaPeriodos, listaGrupos)
        elif mode == 'dataset':
            metodologia.etapa_7_criar_base_dados(listaPeriodos)
        elif mode == 'apredizagem':
            pass
        elif mode == 'exploracao':
            metodologia.etapaexploracaoBaseDados(listaPeriodos)
        elif mode == 'basetexto':
            metodologia.etapaCriarBaseDadosTexto(listaPeriodos, listaGrupos)
        elif mode == 'nuvempalavras':
            metodologia.etapaNuvemPalavras(listaPeriodos)
        elif mode == 'frequencia':
            metodologia.etapaFequenciaPalavras(listaPeriodos, listaGrupos)
        
        else:
            print("Parametro Incorreto")
        
        end_time = datetime.now()
        print('\nTempo de Execucao do modo {}: {}'.format(mode, strfdelta((end_time - start_time))))
    
    except Exception as e:
        print_exc()
        print("Erro critico no programa {}: Exception ".format(mode), e)
    
