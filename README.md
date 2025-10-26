# 🏀 Sistema de Análise de Regressão NBA - Chicago Bulls

Sistema completo para análise preditiva de dados da NBA usando **Regressão Linear** e **Regressão Logística**.  
**Time analisado: Chicago Bulls** | **Temporada 2024-25**

## � Autores

Projeto realizado por [José Daniel](https://github.com/JoseDanielF), [Rian Wilker](https://github.com/RWilker87) e Pedro Medeiros.

## �📋 Descrição

Este sistema baixa dados **REAIS** da NBA usando a `nba_api` e realiza análises estatísticas avançadas para prever:

- **Regressão Linear**: Valores numéricos (pontos, rebotes, assistências)
- **Regressão Logística**: Probabilidades de vitória/derrota

**Análise automatizada pré-configurada para Chicago Bulls com dados da temporada 2024-25 em andamento**

## 🚀 Funcionalidades

### 🎯 MODO AUTOMÁTICO (Novo!)
- ✅ Análise completa dos **Chicago Bulls** em um clique
- ✅ 3 análises de Regressão Linear (Pontos, Rebotes, Assistências)
- ✅ 1 análise de Regressão Logística (Probabilidade de Vitória)
- ✅ 10 gráficos gerados automaticamente
- ✅ 3 cenários de previsão
- ✅ Relatório completo

### 🔧 MODO INTERATIVO
- ✅ Menu completo para análise personalizada
- ✅ Escolha livre de variáveis X e Y
- ✅ Qualquer time da NBA
- ✅ Dados de times e jogadores

## 🛠️ Instalação

### Requisitos
- Python 3.7+
- Windows/Linux/macOS

### Instalar Dependências
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nba_api
```

Ou usar o arquivo de requisitos:
```bash
pip install -r requirements.txt
```

## 📖 Como Usar

### 🚀 Modo Rápido (Recomendado)

```bash
python projeto-p1.py
```

**Escolha a opção 1** para executar análise automática completa dos Chicago Bulls:
- ✅ Baixa dados reais da temporada 2024-25 (ou gera simulação baseada em projeções)
- ✅ Executa 3 Regressões Lineares
- ✅ Executa 1 Regressão Logística
- ✅ Gera todos os 10 gráficos
- ✅ Faz previsões em 3 cenários
- ✅ Apresenta relatório completo

**Tempo de execução: ~30 segundos**

### 🔧 Modo Interativo Personalizado

**Escolha a opção 2** para acesso ao menu completo:

```
SISTEMA DE ANÁLISE DE REGRESSÃO - NBA 2024-25

1. Baixar dados de TIMES
2. Baixar dados de JOGADORES
3. Visualizar dados carregados
4. Executar Regressão LINEAR
5. Gerar gráficos da Regressão Linear
6. Executar Regressão LOGÍSTICA
7. Gerar gráficos da Regressão Logística
8. Fazer previsão de probabilidade de vitória
9. Sair
```

## 📊 Análise Automática Chicago Bulls

### Regressão Linear - 3 Modelos

#### 1. Previsão de PONTOS (PTS)
```
Variável Y: PTS
Variáveis X: FGM, FG3M, FTM
Equação: PTS = β₀ + β₁×FGM + β₂×FG3M + β₃×FTM
```

**Objetivo**: Prever quantos pontos o time fará no jogo

#### 2. Previsão de REBOTES (REB)
```
Variável Y: REB
Variáveis X: FGA, FTA, STL
Equação: REB = β₀ + β₁×FGA + β₂×FTA + β₃×STL
```

**Objetivo**: Prever quantos rebotes o time terá

#### 3. Previsão de ASSISTÊNCIAS (AST)
```
Variável Y: AST
Variáveis X: PTS, FGM, TOV
Equação: AST = β₀ + β₁×PTS + β₂×FGM + β₃×TOV
```

**Objetivo**: Prever quantas assistências o time fará

### Regressão Logística - Probabilidade de Vitória

```
Variável Y: WL (Vitória=1, Derrota=0)
Variáveis X: PTS, FG_PCT, FG3_PCT, REB, AST, STL, TOV
Equação: p = 1 / [1 + e^(-(β₀ + β₁×PTS + β₂×FG_PCT + ... + β₇×TOV))]
```

**Objetivo**: Prever a probabilidade dos Bulls vencerem

### 3 Cenários de Previsão

#### Cenário 1: Estatísticas MÉDIAS
- Baseado no desempenho médio da temporada
- **Pergunta**: "Com desempenho normal, qual a chance de vitória?"

#### Cenário 2: Estatísticas ACIMA da média
- 10% melhor que a média em todas as variáveis
- **Pergunta**: "Em um jogo bom, qual a chance de vitória?"

#### Cenário 3: Estatísticas ABAIXO da média
- 10% pior que a média em todas as variáveis
- **Pergunta**: "Em um jogo ruim, qual a chance de vitória?"

## 📈 Resultados Apresentados

### Métricas de Regressão Linear
- **R²**: Quanto o modelo explica (0 a 1)
- **RMSE**: Erro médio das previsões
- **MAE**: Erro absoluto médio
- **Equação completa**: Com todos os coeficientes β

### Métricas de Regressão Logística
- **Acurácia**: % de previsões corretas
- **Precisão**: Qualidade das previsões de vitória
- **Recall**: % de vitórias corretamente previstas
- **F1-Score**: Média harmônica
- **AUC-ROC**: Qualidade geral do modelo (0.5 a 1.0)
- **Equação completa**: Com todos os coeficientes β

### Exemplo de Saída

```
======================================================================
PREVISÃO DE PROBABILIDADE DE VITÓRIA
======================================================================

   Chicago Bulls vs Golden State Warriors

   Probabilidade de VITÓRIA: 65.23%
   Probabilidade de DERROTA: 34.77%

   PREVISÃO: Chicago Bulls tem 65% de chance de VENCER!
======================================================================
```

## 🎯 Interpretação dos Resultados

### Regressão Linear

#### R² (Coeficiente de Determinação)
- **0.0 - 0.3**: Fraco
- **0.3 - 0.6**: Moderado
- **0.6 - 0.9**: Bom
- **0.9 - 1.0**: Excelente

#### Coeficientes (β)
- **Positivo**: Variável aumenta Y
- **Negativo**: Variável diminui Y
- **Magnitude**: Força do impacto

### Regressão Logística

#### Acurácia
- **< 60%**: Ruim
- **60-70%**: Regular
- **70-80%**: Bom
- **80-90%**: Muito bom
- **> 90%**: Excelente

#### AUC-ROC
- **0.5**: Aleatório (inútil)
- **0.6-0.7**: Fraco
- **0.7-0.8**: Aceitável
- **0.8-0.9**: Bom
- **> 0.9**: Excelente

#### Odds Ratio
- **> 1**: Aumenta chance de vitória
- **< 1**: Diminui chance de vitória
- **= 1**: Sem efeito

## 📁 Arquivos Gerados

Após executar o modo automático, 10 gráficos serão salvos automaticamente em pastas organizadas:

### 📂 Etapa 1 - Regressão Linear (`p1_etapa1_regressao_linear/`)
- `scatter_regression.png` - Dispersão com linha de regressão
- `prediction_vs_reality.png` - Valores previstos vs observados
- `confusion_matrix.png` - Matriz de confusão (binned)
- `trend_confidence.png` - Tendência com intervalo de confiança 95%

### 📂 Etapa 2 - Regressão Logística (`p1_etapa2_regressao_logistica/`)
- `logistic_scatter_sigmoid.png` - Dispersão com curva sigmoide
- `roc_curve.png` - Curva ROC (AUC)
- `predicted_probabilities.png` - Distribuição de probabilidades
- `feature_importance.png` - Importância das variáveis
- `logistic_confusion_matrix.png` - Matriz de confusão
- `logistic_trend_confidence.png` - Tendência com intervalo de confiança

**Nota**: As pastas são criadas automaticamente na primeira execução.

## 📊 Dados Chicago Bulls 2024-25

### Temporada 2024-25 (Em Andamento)

O sistema tenta baixar dados reais da API da NBA. Se a API ainda não tiver dados completos disponíveis, o sistema gera **dados simulados baseados em projeções** para a temporada 2024-25:

- **Record Projetado**: ~38-44 (46% de vitórias)
- **Pontos por jogo**: ~112.5 PPG
- **Field Goal %**: ~46.5%
- **3-Point %**: ~36.2%
- **Free Throw %**: ~80.8%
- **Rebotes por jogo**: ~44.5 RPG
- **Assistências por jogo**: ~26.8 APG
- **Roubos por jogo**: ~7.9 SPG
- **Tocos por jogo**: ~4.6 BPG
- **Turnovers por jogo**: ~13.5 TPG

**Fonte**: Projeções baseadas em histórico recente e início da temporada 2024-25

### ⚠️ Nota Importante
Como a temporada 2024-25 está em andamento (outubro 2024 - abril 2025), os dados podem ser:
1. **Reais** - Se a nba_api já tiver jogos disponíveis
2. **Simulados** - Baseados em estatísticas projetadas realistas

O sistema **tenta automaticamente baixar dados reais** e usa simulação apenas como fallback.

## 🎯 Interpretação dos Resultados

### Regressão Linear

#### R² (Coeficiente de Determinação)
- **0.0 - 0.3**: Fraco - Modelo explica pouco
- **0.3 - 0.6**: Moderado - Modelo tem algum poder preditivo
- **0.6 - 0.9**: Bom - Modelo explica bem os dados
- **0.9 - 1.0**: Excelente - Modelo explica quase tudo

#### Coeficientes (β)
- **β > 0 (Positivo)**: Variável aumenta Y
- **β < 0 (Negativo)**: Variável diminui Y
- **|β| grande**: Forte impacto
- **|β| pequeno**: Fraco impacto

**Exemplo**: 
```
PTS = 10.5 + 2.1×FGM + 3.0×FG3M + 1.0×FTM

Interpretação:
- Cada arremesso normal (FGM) adiciona ~2 pontos
- Cada arremesso de 3 (FG3M) adiciona ~3 pontos  
- Cada lance livre (FTM) adiciona ~1 ponto
```

### Regressão Logística

#### Acurácia (Accuracy)
- **< 60%**: Ruim - Modelo não é confiável
- **60-70%**: Regular - Modelo básico
- **70-80%**: Bom - Modelo confiável
- **80-90%**: Muito bom - Modelo robusto
- **> 90%**: Excelente - Modelo muito preciso

#### AUC-ROC (Area Under Curve)
- **0.5**: Aleatório - Modelo inútil
- **0.6-0.7**: Fraco - Pouco melhor que o acaso
- **0.7-0.8**: Aceitável - Modelo razoável
- **0.8-0.9**: Bom - Modelo discrimina bem
- **> 0.9**: Excelente - Modelo discrimina muito bem

#### Coeficientes e Odds Ratio
- **β > 0**: Aumenta log-odds de vitória
- **β < 0**: Diminui log-odds de vitória
- **e^β > 1**: Aumenta chance de vitória
- **e^β < 1**: Diminui chance de vitória

**Exemplo**:
```
Coeficiente FG_PCT: β = 0.05
Odds Ratio: e^0.05 = 1.05

Interpretação:
- Cada 1% a mais em FG% aumenta as chances de vitória em 5%
```

## 🔧 Solução de Problemas

### Erro ao baixar dados da API
```
⚠️ Problema: API não retorna dados
✅ Solução: Sistema gera dados baseados em estatísticas reais automaticamente
```

### "Dados insuficientes para análise"
```
⚠️ Problema: Menos de 10 amostras
✅ Solução: Use dados de times (82 jogos) ao invés de jogadores
```

### Variável WL não encontrada
```
⚠️ Problema: Coluna WL ausente
✅ Solução: Sistema cria automaticamente baseado em PLUS_MINUS > 0
```

### Gráficos não aparecem
```
⚠️ Problema: Matplotlib não exibe janelas
✅ Solução: Gráficos são salvos como PNG no diretório atual
```

### nba_api muito lenta
```
⚠️ Problema: Downloads demoram muito
✅ Solução: Sistema usa dados de exemplo após timeout
```

## 📚 Estrutura do Projeto

```
projeto redes neurais/
│
├── projeto-p1.py              # Código principal
├── requirements.txt           # Dependências
├── README.md                  # Este arquivo
│
├── p1_etapa1_regressao_linear/  # Gráficos da Etapa 1
│   ├── scatter_regression.png
│   ├── prediction_vs_reality.png
│   ├── confusion_matrix.png
│   └── trend_confidence.png
│
└── p1_etapa2_regressao_logistica/  # Gráficos da Etapa 2
    ├── logistic_scatter_sigmoid.png
    ├── roc_curve.png
    ├── predicted_probabilities.png
    ├── feature_importance.png
    ├── logistic_confusion_matrix.png
    └── logistic_trend_confidence.png
```

## 🎓 Conceitos Implementados

### Machine Learning
- ✅ Regressão Linear simples e múltipla
- ✅ Regressão Logística binária
- ✅ Train/Test split (80/20)
- ✅ Validação cruzada
- ✅ Métricas de avaliação completas

### Estatística
- ✅ Coeficientes de correlação
- ✅ Intervalos de confiança (95%)
- ✅ P-valores implícitos
- ✅ Odds ratios
- ✅ Curva ROC e AUC

### Visualização
- ✅ Matplotlib para gráficos
- ✅ Seaborn para estilização
- ✅ 10 tipos diferentes de gráficos
- ✅ Alta qualidade (300 DPI)

## 📚 Referências

- [nba_api Documentation](https://github.com/swar/nba_api)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NBA Official Stats](https://www.nba.com/stats)
- [Chicago Bulls Official](https://www.nba.com/bulls)
- [Basketball Reference](https://www.basketball-reference.com/)

## ✅ Requisitos Atendidos

### Parte 1 - Regressão Linear
- ✅ Download via nba_api
- ✅ Time: Chicago Bulls
- ✅ Interface para escolha de variáveis X e Y
- ✅ Sistema genérico e flexível
- ✅ Equação completa: y = β₀ + β₁x₁ + ... + βₙxₙ + ε
- ✅ Quantificação de impacto (coeficientes β)
- ✅ Previsão de valores futuros
- ✅ Regressão múltipla
- ✅ Todos os 4 gráficos obrigatórios
- ✅ Hipóteses: Pontos, Rebotes, Assistências

### Parte 2 - Regressão Logística
- ✅ Curva Sigmoide: p = 1/[1 + e^-(z)]
- ✅ Probabilidades entre 0 e 1
- ✅ Interpretação > 0.5 = Vitória
- ✅ Variável Y: Vitória (1) / Derrota (0)
- ✅ Interface para escolha de variáveis X
- ✅ Probabilidades percentuais
- ✅ Exemplo: "Bulls têm 65% de chance de vencer"
- ✅ Todos os 6 gráficos obrigatórios

## 👨‍💻 Desenvolvimento

**Linguagem**: Python 3.13  
**Framework ML**: Scikit-learn  
**API**: nba_api  
**Visualização**: Matplotlib + Seaborn  

## 📄 Licença

Este projeto é de código aberto para fins educacionais.

---

**Desenvolvido para análise preditiva da NBA** 🏀📊

**Chicago Bulls - Temporada 2024-25** 🔴⚫
