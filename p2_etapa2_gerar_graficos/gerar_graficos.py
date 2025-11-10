import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import os

# Definir o estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_prepare_data(features, target='WL', test_size=0.15, val_size=0.15):
    """
    Recria a divisão de dados EXATA usada no script mlp_completo_chicago_bulls.py
    para isolar o conjunto de teste.
    """
    data_file = '../chicago_bulls_2024_25_games.csv'
    if not os.path.exists(data_file):
        print(f"Erro: Arquivo de dados não encontrado: {data_file}")
        return None, None, None, None, None

    data = pd.read_csv(data_file)

    # Limpar dados como no script da MLP
    data_clean = data.copy()
    data_clean['WL_binary'] = (data_clean[target] == 'W').astype(int)
    data_clean = data_clean[features + ['WL_binary']].dropna()
    data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()

    X = data_clean[features].values
    y = data_clean['WL_binary'].values

    # 1. Separar o conjunto de TESTE (15%) - O MESMO random_state
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 2. Separar Treino (70%) e Validação (15%) - O MESMO random_state
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )

    # 3. Normalizar os dados (ESSENCIAL)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Retorna os conjuntos que importam para a comparação
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_mlp_predictions():
    """
    Carrega as previsões já calculadas do melhor modelo MLP (Adam).
    """
    pred_file = '../p2_etapa1_relatorio/resultados_mlp/previsao_realidade/previsao_realidade_adam.csv'
    if not os.path.exists(pred_file):
        print(f"Erro: Arquivo de previsões da MLP não encontrado: {pred_file}")
        return None, None

    try:
        df_mlp = pd.read_csv(pred_file)
        # Converte a prob de string '88.11%' para float 0.8811
        df_mlp['Prob_Vitoria_Float'] = df_mlp['Prob_Vitoria'].str.replace('%', '').astype(float) / 100.0

        # Ordenar por Jogo para garantir a ordem correta
        df_mlp = df_mlp.sort_values(by='Jogo')

        y_pred_proba_mlp = df_mlp['Prob_Vitoria_Float'].values
        y_pred_mlp = (y_pred_proba_mlp > 0.5).astype(int)

        return y_pred_mlp, y_pred_proba_mlp
    except Exception as e:
        print(f"Erro ao processar arquivo {pred_file}: {e}")
        return None, None


def train_and_predict_logistic_regression(X_train_scaled, X_test_scaled, y_train):
    """
    Treina o modelo de Regressão Logística SOMENTE nos dados de treino da MLP
    e prevê o conjunto de teste.
    """
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_test_scaled)
    y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]  # Probabilidade de Vitória (classe 1)

    return y_pred_lr, y_pred_proba_lr


def create_comparison_metrics(y_test, y_pred_mlp, y_pred_proba_mlp, y_pred_lr, y_pred_proba_lr):
    """
    Calcula e salva as métricas de comparação.
    """
    metrics = {
        'Metrica': ['Acuracia', 'Precisao', 'F1-Score', 'AUC-ROC'],
        'MLP (Adam)': [
            accuracy_score(y_test, y_pred_mlp),
            precision_score(y_test, y_pred_mlp, zero_division=0),
            f1_score(y_test, y_pred_mlp, zero_division=0),
            roc_auc_score(y_test, y_pred_proba_mlp)
        ],
        'Reg. Logistica (Atv 1)': [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr, zero_division=0),
            f1_score(y_test, y_pred_lr, zero_division=0),
            roc_auc_score(y_test, y_pred_proba_lr)
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('comparacao_metricas.csv', index=False)
    print("\nArquivo 'comparacao_metricas.csv' gerado com sucesso.")
    print(df_metrics.to_string())
    return df_metrics


def create_comparison_plot(y_test, y_pred_proba_mlp, y_pred_proba_lr):
    """
    Gera o gráfico de barras comparativo "lado a lado".
    """
    df_plot = pd.DataFrame({
        'Jogo': [f'Jogo {i + 1}' for i in range(len(y_test))],
        'Real': y_test,
        'Prob_MLP': y_pred_proba_mlp,
        'Prob_LogReg': y_pred_proba_lr
    })

    # Reformatar para plotagem
    df_melted = df_plot.melt(id_vars=['Jogo', 'Real'],
                             value_vars=['Prob_MLP', 'Prob_LogReg'],
                             var_name='Modelo',
                             value_name='Probabilidade')

    plt.figure(figsize=(15, 8))

    # Gráfico de barras agrupadas
    sns.barplot(data=df_melted, x='Jogo', y='Probabilidade', hue='Modelo',
                palette={'Prob_MLP': 'blue', 'Prob_LogReg': 'green'},
                edgecolor='black', alpha=0.8)

    # Linha de threshold
    plt.axhline(0.5, ls='--', color='red', label='Limiar de Decisão (0.5)')

    plt.title('Comparação Lado a Lado: MLP vs Regressão Logística (Conjunto de Teste)',
              fontsize=16, fontweight='bold')
    plt.ylabel('Probabilidade de Vitória', fontsize=12)
    plt.xlabel('Jogos do Conjunto de Teste', fontsize=12)
    plt.legend(title='Modelo')

    # Colorir os ticks do eixo X com base no resultado real
    # Verde para Vitória (1), Vermelho para Derrota (0)
    ax = plt.gca()
    jogo_labels = sorted(df_plot['Jogo'].unique())
    real_results = df_plot.drop_duplicates(subset=['Jogo']).sort_values(by='Jogo')['Real'].values

    for tick in ax.get_xticklabels():
        jogo_name = tick.get_text()
        try:
            # Encontrar o índice do jogo
            idx = jogo_labels.index(jogo_name)
            if real_results[idx] == 1:  # Vitória
                tick.set_color('green')
                tick.set_fontweight('bold')
            else:  # Derrota
                tick.set_color('red')
                tick.set_fontweight('bold')
        except ValueError:
            pass  # Ignorar se o label não for encontrado

    plt.tight_layout()
    plt.savefig('comparacao_mlp_vs_logreg.png', dpi=300)
    print("\nGráfico 'comparacao_mlp_vs_logreg.png' gerado com sucesso.")
    plt.close()


# --- Execução Principal ---
print("Iniciando Atividade 2 - Parte 2: Comparação MLP vs Reg. Logística")

# 1. Definir features (as mesmas do Relatório e script MLP)
features = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'TOV']

# 2. Recriar os conjuntos de dados
X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data(features)

if y_test is not None:
    print(f"\nConjunto de dados recriado com sucesso.")
    print(f"  - Amostras de Treino: {len(y_train)}")
    print(f"  - Amostras de Teste: {len(y_test)} (Este é o conjunto de comparação)")

    # 3. Carregar previsões da MLP (Adam)
    y_pred_mlp, y_pred_proba_mlp = get_mlp_predictions()

    if y_pred_mlp is not None:
        print("Previsões da MLP (Adam) carregadas com sucesso.")

        # 4. Treinar e prever com Regressão Logística
        y_pred_lr, y_pred_proba_lr = train_and_predict_logistic_regression(
            X_train_scaled, X_test_scaled, y_train
        )
        print("Modelo de Regressão Logística treinado e avaliado no mesmo conjunto.")

        # 5. Gerar Tabela de Métricas
        create_comparison_metrics(y_test, y_pred_mlp, y_pred_proba_mlp, y_pred_lr, y_pred_proba_lr)

        # 6. Gerar Gráfico Comparativo
        create_comparison_plot(y_test, y_pred_proba_mlp, y_pred_proba_lr)

        print("\n--- Concluído ---")
        print("Verifique os arquivos 'comparacao_metricas.csv' e 'comparacao_mlp_vs_logreg.png'.")
    else:
        print("Não foi possível carregar as previsões da MLP. Encerrando.")
else:
    print("Não foi possível carregar os dados. Encerrando.")