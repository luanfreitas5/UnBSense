# UnBSense

## Requirements

- tweepy==3.10.0
- snscrape
- pandas
- numpy
- tqdm
- unidecode
- nltk
- spacy
- matplotlib
- sklearn
- xlrd
- openpyxl
- seaborn
- matplotlib_venn
- wordcloud
- jupyter

## Installing

**Git:**
```bash
git clone https://github.com/luanfreitas5/UnBSense.git
cd twint
pip install -r requirements.txt
```

## Etapas para executar 

1. Buscar Usuarios (depressivos e não depressivos) no Twitter - Multi-Threads
    ```bash    
    python main.py -m buscar
    ```

2. Coletar tweets de Usuarios (depressivos e não depressivos) no Twitter - Multi-Threads  (processo demorado)
    ```bash    
    python main.py -m coletar
    ```
    
3. Mesclar datasets de tweets - Uma Thread
    ```bash    
    python main.py -m mesclar
    ```
    
4. Preparação de Textos nos tweets - Uma Thread
    ```bash    
    python main.py -m limpar
    ```
    
5. Extração de caracteristicas (atributos) - Uma Thread
    ```bash    
    python main.py -m atributos
    ```
    
6. Limpeza de Outliens - Uma Thread
    ```bash    
    python main.py -m qualidade
    ```
    
7. Calculo do vetores de caracteristicas - Uma Thread
    ```bash    
    python main.py -m vetores
    ```
    
8. Criar bases de dados
    ```bash    
    python main.py -m dataset
    ```
    
9. Plotar graicos de exploração de dados - Uma Thread
    ```bash    
    python main.py -m exploracao
    ```
    
9. Plotar graficos de exploração de dados - Uma Thread
    ```bash    
    python main.py -m exploracao
    ```
    
10. Criar base de textos para nuvem de palavras - Multi-Thread (processo demorado)
    ```bash    
    python main.py -m basetexto
    ```
    
11. Obter frequencia de palavras para nuvem de palavras - Multi-Thread
    ```bash    
    python main.py -m basetexto
    ```
    
12. Plotar nuvem de palavras - Uma Thread
    ```bash    
    python main.py -m nuvempalavras
    ```