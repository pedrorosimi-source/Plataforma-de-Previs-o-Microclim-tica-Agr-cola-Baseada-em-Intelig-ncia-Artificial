### Arquitetura Técnica e Módulos

1. **Tratamento e Engenharia de Dados (`pipeline.py`)**:
   - **Interpolação por Spline Cúbica (`CubicSpline`)**: Preenche falhas e lacunas na série temporal de variáveis microclimáticas (Precipitação, Temperatura do Solo e Humidade Relativa). 
   - **Normalização MinMax**: Converte os dados brutos para o intervalo [0, 1], estabilizando a convergência matemática dos algoritmos.
   - **Janela Deslizante**: Converte séries temporais contínuas em tensores estruturados (X, y) para aprendizagem supervisionada baseada na janela histórica e no horizonte preditivo.

2. **Motor de IA Customizado (`model.py`)**:
   - Implementa uma **Célula LSTM construída de raiz com `numpy`, manipulando vetores matriciais para as portas de esquecimento, entrada, candidato a estado de célula e saída.
   - Otimização numérica através da função de perda **MSE** integrada com **Regularização de Tikhonov**, prevenindo *overfitting* na previsão microclimática.

3. **Interface e Orquestração (`principal.py` & `config.yaml`)**:
   - Painel interativo construído em **Streamlit**.
   - Parâmetros dinâmicos configuráveis pela barra lateral (Tamanho da Janela, Horizonte de Previsão, Dimensão Oculta, Épocas, Penalização e Paciência do *Early Stopping*).
   - Visualização gráfica de curvas de aprendizagem, aderência histórica (*In-Sample/Out-of-Sample*) e tabela preditiva do microclima.
   - **Gerador de Diretrizes Agrícolas**: Recomendações diárias automáticas de irrigação com base na humidade relativa e precipitação previstas.

---

### Instalação das Bibliotecas Necessárias

Para instalar todas as bibliotecas necessárias, abra o terminal e execute o seguinte comando:

```bash
pip install numpy pandas scipy matplotlib seaborn pyyaml streamlit scikit-learn

1. numpy: biblioteca utilizada para operações matriciais e álgebra linear da LSTM.
2. scipy: utilizada para interpolação matemática por Splines Cúbicas.
3. matplotlib & seaborn: Renderização dos gráficos de desempenho e tendência microclimática.
4. pyyaml: Leitura de configurações dinâmicas a partir de arquivos .yaml.
5. streamlit: Framework da interface gráfica interativa web.
6. scikit-learn: Métricas de avaliação e modelos de regressão para benchmark.

streamlit run principal.py: # Para colocar o sistema a rodar, abra o terminal e digite streamlit run principal.py e clicar enter.
