# -*- coding: utf-8 -*-
"""
Atividade 2 - MLP para Previsao de Vitorias/Derrotas - Chicago Bulls
Sistema Completo com TODOS os graficos solicitados
Temporada NBA 2024-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
from scipy import stats

warnings.filterwarnings('ignore')

# Configurar seed para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Criar pastas para salvar resultados
os.makedirs('resultados_mlp/graficos_treinamento', exist_ok=True)
os.makedirs('resultados_mlp/matriz_erros', exist_ok=True)
os.makedirs('resultados_mlp/previsao_realidade', exist_ok=True)
os.makedirs('resultados_mlp/rankings', exist_ok=True)
os.makedirs('resultados_mlp/comparacoes', exist_ok=True)
os.makedirs('resultados_mlp/intervalos_confianca', exist_ok=True)

# Configuracao de visualizacao
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLPNBACompleto:
    """
    Classe COMPLETA para previsao de vitorias/derrotas usando MLP
    Inclui TODOS os graficos solicitados na atividade
    """
    
    def __init__(self, data):
        self.data = data
        self.model = None
        self.history = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_names = None
        
    def prepare_data(self, features, target='WL', test_size=0.15, val_size=0.15):
        """Prepara os dados para treinamento da MLP"""
        print("="*80)
        print("PREPARACAO DOS DADOS")
        print("="*80)
        
        self.selected_features = features
        self.feature_names = features
        
        # Converter WL para binario: W=1, L=0
        data_clean = self.data.copy()
        data_clean['WL_binary'] = (data_clean[target] == 'W').astype(int)
        
        # Remover valores nulos e infinitos
        data_clean = data_clean[features + ['WL_binary']].dropna()
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"\nTotal de amostras validas: {len(data_clean)}")
        print(f"  - Vitorias: {sum(data_clean['WL_binary'] == 1)} ({sum(data_clean['WL_binary'] == 1)/len(data_clean)*100:.1f}%)")
        print(f"  - Derrotas: {sum(data_clean['WL_binary'] == 0)} ({sum(data_clean['WL_binary'] == 0)/len(data_clean)*100:.1f}%)")
        
        # Separar features (X) e target (y)
        X = data_clean[features].values
        y = data_clean['WL_binary'].values
        
        # Dividir em treino, validacao e teste
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Normalizacao dos dados
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nDivisao dos dados:")
        print(f"  - Treinamento: {len(self.X_train)} amostras ({len(self.X_train)/len(data_clean)*100:.1f}%)")
        print(f"  - Validacao: {len(self.X_val)} amostras ({len(self.X_val)/len(data_clean)*100:.1f}%)")
        print(f"  - Teste: {len(self.X_test)} amostras ({len(self.X_test)/len(data_clean)*100:.1f}%)")
        
        return self
    
    def build_model(self, hidden_layers=[32, 16], activation='relu', 
                   dropout_rate=0.3, use_batch_norm=True):
        """Constroi a arquitetura da MLP"""
        print("\n" + "="*80)
        print("ARQUITETURA DA MLP")
        print("="*80)
        
        input_dim = self.X_train.shape[1]
        
        self.model = Sequential()
        
        # Camada de Entrada + Primeira Camada Oculta
        self.model.add(Dense(hidden_layers[0], activation=activation, 
                            input_dim=input_dim, name='entrada_oculta1'))
        
        if use_batch_norm:
            self.model.add(BatchNormalization())
        
        self.model.add(Dropout(dropout_rate))
        
        # Camadas Ocultas Intermediarias
        for i, neurons in enumerate(hidden_layers[1:], 2):
            self.model.add(Dense(neurons, activation=activation, 
                               name=f'oculta{i}'))
            if use_batch_norm:
                self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout_rate))
        
        # Camada de Saida
        self.model.add(Dense(1, activation='sigmoid', name='saida'))
        
        print(f"\nConfiguracao da rede:")
        print(f"  - Camada de Entrada: {input_dim} neuronios")
        print(f"  - Camadas Ocultas: {len(hidden_layers)} camadas")
        for i, neurons in enumerate(hidden_layers, 1):
            print(f"    Camada Oculta {i}: {neurons} neuronios")
        print(f"  - Camada de Saida: 1 neuronio (Vitoria/Derrota)")
        print(f"  - Funcao de Ativacao: {activation}")
        print(f"  - Dropout Rate: {dropout_rate}")
        print(f"  - Batch Normalization: {'Sim' if use_batch_norm else 'Nao'}")
        
        return self
    
    def train_model(self, optimizer_name='adam', epochs=200, batch_size=16,
                   early_stopping=True, patience=20, verbose=1):
        """Treina o modelo MLP"""
        print("\n" + "="*80)
        print(f"TREINAMENTO - Otimizador: {optimizer_name.upper()}")
        print("="*80)
        
        # Selecionar otimizador
        if optimizer_name.lower() == 'sgd':
            optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        elif optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=0.001)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=0.001)
        else:
            raise ValueError(f"Otimizador desconhecido: {optimizer_name}")
        
        # Compilar modelo
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nConfiguracao do treinamento:")
        print(f"  - Otimizador: {optimizer_name.upper()}")
        print(f"  - Funcao de Perda: Binary Crossentropy")
        print(f"  - Epocas: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Early Stopping: {'Sim' if early_stopping else 'Nao'}")
        if early_stopping:
            print(f"  - Patience: {patience} epocas")
        
        # Callbacks
        callbacks = []
        if early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(es)
        
        # Treinar modelo
        print("\nIniciando treinamento...\n")
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\nTreinamento concluido!")
        
        return self
    
    def evaluate_model(self):
        """Avalia o desempenho do modelo"""
        print("\n" + "="*80)
        print("AVALIACAO DO MODELO")
        print("="*80)
        
        # Predicoes
        y_train_pred = (self.model.predict(self.X_train, verbose=0) > 0.5).astype(int)
        y_val_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
        y_test_pred = (self.model.predict(self.X_test, verbose=0) > 0.5).astype(int)
        
        # Metricas para conjunto de teste
        print("\nCONJUNTO DE TESTE:")
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_prec = precision_score(self.y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(self.y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)
        
        print(f"  - Acuracia: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  - Precisao: {test_prec:.4f}")
        print(f"  - Recall: {test_rec:.4f}")
        print(f"  - F1-Score: {test_f1:.4f}")
        
        # Matriz de Confusao
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\nMatriz de Confusao:\n{cm}")
        
        # Metricas de Regressao (para probabilidades)
        y_test_prob = self.model.predict(self.X_test, verbose=0)
        mae = mean_absolute_error(self.y_test, y_test_prob)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_test_prob))
        r2 = r2_score(self.y_test, y_test_prob)
        
        print(f"\nMetricas de Regressao (Probabilidades):")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R2 Score: {r2:.4f}")
        
        return {
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'confusion_matrix': cm
        }
    
    # ==================== GRAFICOS SOLICITADOS ====================
    
    def plot_evolucao_erro_treinamento(self, optimizer_name):
        """
        GRAFICO 1: Evolucao do Erro durante Treinamento
        Mostra erro em treino e validacao
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss (Erro)
        epochs_range = range(1, len(self.history.history['loss']) + 1)
        axes[0].plot(epochs_range, self.history.history['loss'], 
                    label='Treino', linewidth=2, marker='o', markersize=3)
        axes[0].plot(epochs_range, self.history.history['val_loss'], 
                    label='Validacao', linewidth=2, marker='s', markersize=3)
        axes[0].set_title(f'Evolucao do Erro - {optimizer_name.upper()}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoca', fontsize=12)
        axes[0].set_ylabel('Erro (Binary Crossentropy)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs_range, self.history.history['accuracy'], 
                    label='Treino', linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs_range, self.history.history['val_accuracy'], 
                    label='Validacao', linewidth=2, marker='s', markersize=3)
        axes[1].set_title(f'Acuracia - {optimizer_name.upper()}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoca', fontsize=12)
        axes[1].set_ylabel('Acuracia', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'resultados_mlp/graficos_treinamento/evolucao_erro_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: evolucao_erro_{optimizer_name}.png")
        plt.close()
    
    def plot_matriz_erros_completa(self, optimizer_name):
        """
        GRAFICO 2: Matriz de Erros (3 graficos)
        HISTOGRAMA + SCATTER PLOT + SCATTER PLOT
        """
        y_test_prob = self.model.predict(self.X_test, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)
        erros = self.y_test - y_test_prob
        
        fig = plt.figure(figsize=(18, 5))
        
        # HISTOGRAMA dos erros
        ax1 = plt.subplot(1, 3, 1)
        ax1.hist(erros, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
        ax1.set_title('HISTOGRAMA - Distribuicao dos Erros', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Erro (Real - Previsto)', fontsize=10)
        ax1.set_ylabel('Frequencia', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SCATTER PLOT 1: Previsto vs Real
        ax2 = plt.subplot(1, 3, 2)
        cores = ['red' if real != pred else 'green' 
                for real, pred in zip(self.y_test, y_test_pred)]
        ax2.scatter(self.y_test, y_test_prob, c=cores, alpha=0.6, s=100, edgecolor='black')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Previsao Perfeita')
        ax2.set_title('SCATTER - Previsto vs Real', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Valor Real', fontsize=10)
        ax2.set_ylabel('Valor Previsto (Probabilidade)', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # SCATTER PLOT 2: Erro vs Indice da Amostra
        ax3 = plt.subplot(1, 3, 3)
        indices = np.arange(len(erros))
        cores2 = ['red' if abs(e) > 0.3 else 'green' for e in erros]
        ax3.scatter(indices, erros, c=cores2, alpha=0.6, s=100, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax3.set_title('SCATTER - Erro por Amostra', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Indice da Amostra', fontsize=10)
        ax3.set_ylabel('Erro', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'resultados_mlp/matriz_erros/matriz_erros_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: matriz_erros_{optimizer_name}.png")
        plt.close()
    
    def plot_previsao_vs_realidade(self, optimizer_name):
        """
        GRAFICO 3: Previsao x Realidade
        Tabela comparativa mostrando previsto vs real
        """
        y_test_prob = self.model.predict(self.X_test, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)
        
        # Criar DataFrame comparativo
        df_comparacao = pd.DataFrame({
            'Jogo': [f'Jogo {i+1}' for i in range(len(self.y_test))],
            'Real': ['Vitoria' if y == 1 else 'Derrota' for y in self.y_test],
            'Previsto': ['Vitoria' if y == 1 else 'Derrota' for y in y_test_pred],
            'Prob_Vitoria': [f'{p:.2%}' for p in y_test_prob],
            'Correto': ['?' if r == p else '?' 
                       for r, p in zip(self.y_test, y_test_pred)]
        })
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Criar tabela
        table = ax.table(cellText=df_comparacao.values,
                        colLabels=df_comparacao.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.2, 0.2, 0.25, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorir header
        for i in range(len(df_comparacao.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorir linhas alternadas e marcar erros
        for i in range(1, len(df_comparacao) + 1):
            for j in range(len(df_comparacao.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                
                # Marcar erros em vermelho
                if df_comparacao.iloc[i-1]['Correto'] == '?':
                    table[(i, j)].set_facecolor('#ffcccc')
        
        plt.title(f'Previsao x Realidade - {optimizer_name.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(f'resultados_mlp/previsao_realidade/previsao_realidade_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: previsao_realidade_{optimizer_name}.png")
        plt.close()
        
        # Salvar CSV tambem
        df_comparacao.to_csv(f'resultados_mlp/previsao_realidade/previsao_realidade_{optimizer_name}.csv', 
                            index=False)
    
    def plot_ranking_previsoes(self, optimizer_name):
        """
        GRAFICO 4: Ranking de Previsoes
        Ranking dos jogos com maiores probabilidades de vitoria
        """
        y_test_prob = self.model.predict(self.X_test, verbose=0).flatten()
        
        # Criar DataFrame com probabilidades
        df_ranking = pd.DataFrame({
            'Jogo': [f'Jogo {i+1}' for i in range(len(self.y_test))],
            'Prob_Vitoria': y_test_prob,
            'Real': self.y_test
        })
        
        # Ordenar por probabilidade
        df_ranking = df_ranking.sort_values('Prob_Vitoria', ascending=False)
        df_ranking['Ranking'] = range(1, len(df_ranking) + 1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cores = ['green' if r == 1 else 'red' for r in df_ranking['Real']]
        bars = ax.barh(df_ranking['Jogo'], df_ranking['Prob_Vitoria'], color=cores, alpha=0.7)
        
        ax.set_xlabel('Probabilidade de Vitoria', fontsize=12)
        ax.set_ylabel('Jogo', fontsize=12)
        ax.set_title(f'Ranking de Previsoes - {optimizer_name.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Real: Vitoria'),
                          Patch(facecolor='red', alpha=0.7, label='Real: Derrota')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'resultados_mlp/rankings/ranking_previsoes_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: ranking_previsoes_{optimizer_name}.png")
        plt.close()
        
        # Salvar CSV
        df_ranking.to_csv(f'resultados_mlp/rankings/ranking_previsoes_{optimizer_name}.csv', 
                         index=False)
    
    def plot_intervalo_confianca(self, optimizer_name):
        """
        GRAFICO 5: Intervalos de Confianca
        Mostra intervalos de confianca para as previsoes
        """
        y_test_prob = self.model.predict(self.X_test, verbose=0).flatten()
        
        # Calcular intervalos de confianca (95%)
        confidence = 0.95
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Estimativa do desvio padrao baseado em probabilidade binomial
        std_errors = np.sqrt(y_test_prob * (1 - y_test_prob) / len(self.y_test))
        margin_of_error = z_score * std_errors
        
        lower_bound = np.maximum(0, y_test_prob - margin_of_error)
        upper_bound = np.minimum(1, y_test_prob + margin_of_error)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(y_test_prob))
        cores = ['green' if r == 1 else 'red' for r in self.y_test]
        
        # Pontos previstos
        ax.scatter(x, y_test_prob, c=cores, s=100, alpha=0.7, 
                  edgecolor='black', linewidth=1, label='Previsao', zorder=3)
        
        # Intervalos de confianca
        ax.errorbar(x, y_test_prob, 
                   yerr=[y_test_prob - lower_bound, upper_bound - y_test_prob],
                   fmt='none', ecolor='gray', alpha=0.5, capsize=3, zorder=2)
        
        # Linha de referencia
        ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, 
                  label='Limiar (0.5)', zorder=1)
        
        ax.set_xlabel('Indice do Jogo', fontsize=12)
        ax.set_ylabel('Probabilidade de Vitoria', fontsize=12)
        ax.set_title(f'Intervalos de Confianca (95%) - {optimizer_name.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Real: Vitoria'),
            Patch(facecolor='red', alpha=0.7, label='Real: Derrota'),
            plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Limiar (0.5)')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'resultados_mlp/intervalos_confianca/intervalos_confianca_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: intervalos_confianca_{optimizer_name}.png")
        plt.close()
    
    def plot_evolucao_temporal(self, optimizer_name):
        """
        GRAFICO 6: Evolucao Temporal
        Mostra como as previsoes evoluem ao longo do tempo/jogos
        """
        y_test_prob = self.model.predict(self.X_test, verbose=0).flatten()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Grafico 1: Probabilidades ao longo do tempo
        x = np.arange(len(y_test_prob))
        cores = ['green' if r == 1 else 'red' for r in self.y_test]
        
        axes[0].plot(x, y_test_prob, 'b-', linewidth=2, alpha=0.7, label='Prob. Prevista')
        axes[0].scatter(x, y_test_prob, c=cores, s=100, alpha=0.8, edgecolor='black', linewidth=1)
        axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Limiar')
        axes[0].fill_between(x, 0, y_test_prob, where=(y_test_prob >= 0.5), 
                            alpha=0.2, color='green', label='Zona Vitoria')
        axes[0].fill_between(x, 0, y_test_prob, where=(y_test_prob < 0.5), 
                            alpha=0.2, color='red', label='Zona Derrota')
        axes[0].set_xlabel('Sequencia de Jogos', fontsize=12)
        axes[0].set_ylabel('Probabilidade de Vitoria', fontsize=12)
        axes[0].set_title(f'Evolucao Temporal das Previsoes - {optimizer_name.upper()}', 
                         fontsize=13, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.05, 1.05)
        
        # Grafico 2: Media movel das probabilidades
        window = 3  # Janela de 3 jogos
        if len(y_test_prob) >= window:
            moving_avg = pd.Series(y_test_prob).rolling(window=window).mean()
            axes[1].plot(x, y_test_prob, 'o-', alpha=0.3, color='gray', label='Prob. Individual')
            axes[1].plot(x, moving_avg, 'b-', linewidth=3, label=f'Media Movel ({window} jogos)')
            axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Limiar')
            axes[1].set_xlabel('Sequencia de Jogos', fontsize=12)
            axes[1].set_ylabel('Probabilidade de Vitoria', fontsize=12)
            axes[1].set_title(f'Media Movel das Previsoes - {optimizer_name.upper()}', 
                             fontsize=13, fontweight='bold')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(f'resultados_mlp/comparacoes/evolucao_temporal_{optimizer_name}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"? Grafico salvo: evolucao_temporal_{optimizer_name}.png")
        plt.close()
    
    def gerar_todos_graficos(self, optimizer_name):
        """Gera TODOS os graficos solicitados"""
        print("\n" + "="*80)
        print("GERANDO TODOS OS GRAFICOS")
        print("="*80)
        
        self.plot_evolucao_erro_treinamento(optimizer_name)
        self.plot_matriz_erros_completa(optimizer_name)
        self.plot_previsao_vs_realidade(optimizer_name)
        self.plot_ranking_previsoes(optimizer_name)
        self.plot_intervalo_confianca(optimizer_name)
        self.plot_evolucao_temporal(optimizer_name)
        
        print("\n? TODOS os graficos foram gerados com sucesso!")


def comparar_otimizadores(data, features):
    """Compara os 3 otimizadores: SGD, Adam, RMSprop"""
    print("\n" + "="*80)
    print("COMPARACAO DOS 3 OTIMIZADORES")
    print("="*80)
    
    optimizers = ['sgd', 'adam', 'rmsprop']
    results = {}
    
    for optimizer_name in optimizers:
        print(f"\n{'='*80}")
        print(f"TESTANDO: {optimizer_name.upper()}")
        print(f"{'='*80}")
        
        # Criar e treinar modelo
        mlp = MLPNBACompleto(data)
        mlp.prepare_data(features)
        mlp.build_model(hidden_layers=[32, 16], activation='relu', 
                       dropout_rate=0.3, use_batch_norm=True)
        mlp.train_model(optimizer_name=optimizer_name, epochs=200, 
                       batch_size=16, early_stopping=True, patience=20, verbose=0)
        
        # Avaliar
        metrics = mlp.evaluate_model()
        results[optimizer_name] = metrics
        
        # Gerar TODOS os graficos
        mlp.gerar_todos_graficos(optimizer_name)
    
    # Comparacao final
    print("\n" + "="*80)
    print("RESUMO COMPARATIVO")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Otimizador': list(results.keys()),
        'Acuracia': [f"{results[opt]['test_acc']:.4f}" for opt in results.keys()],
        'Precisao': [f"{results[opt]['test_prec']:.4f}" for opt in results.keys()],
        'F1-Score': [f"{results[opt]['test_f1']:.4f}" for opt in results.keys()],
        'MAE': [f"{results[opt]['mae']:.4f}" for opt in results.keys()],
        'RMSE': [f"{results[opt]['rmse']:.4f}" for opt in results.keys()],
        'R2': [f"{results[opt]['r2']:.4f}" for opt in results.keys()]
    })
    
    print("\n", comparison_df.to_string(index=False))
    
    # Salvar comparacao
    comparison_df.to_csv('resultados_mlp/comparacao_otimizadores.csv', index=False)
    print("\n? Comparacao salva em: comparacao_otimizadores.csv")
    
    return results


def main():
    """Funcao principal - Execucao completa"""
    print("\n" + "="*80)
    print("ATIVIDADE 2 - MLP CHICAGO BULLS 2024-25")
    print("Sistema COMPLETO com TODOS os graficos")
    print("="*80)
    
    # Carregar dados
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(parent_dir, 'chicago_bulls_2024_25_games.csv')
    
    if not os.path.exists(csv_file):
        print(f"\nERRO: Arquivo nao encontrado: {csv_file}")
        return
    
    print(f"\nCarregando dados: {csv_file}")
    data = pd.read_csv(csv_file)
    print(f"? {len(data)} jogos carregados")
    
    # Features selecionadas (baseadas na Atividade 1)
    features = ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'TOV']
    
    print("\nFeatures selecionadas (baseadas na Atividade 1):")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    # Executar comparacao completa
    results = comparar_otimizadores(data, features)
    
    print("\n" + "="*80)
    print("ANALISE CONCLUIDA COM SUCESSO!")
    print("="*80)
    print("\nTodos os resultados foram salvos em: resultados_mlp/")
    print("  ? graficos_treinamento/ - Evolucao do erro")
    print("  ? matriz_erros/ - Histograma + Scatter plots")
    print("  ? previsao_realidade/ - Tabelas comparativas")
    print("  ? rankings/ - Rankings de previsoes")
    print("  ? intervalos_confianca/ - Intervalos de confianca")
    print("  ? comparacoes/ - Evolucao temporal")
    print("="*80)


if __name__ == "__main__":
    main()
