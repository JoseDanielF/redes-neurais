"""
Sistema de Análise de Regressão Linear para NBA
Temporada 2024-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Criar pastas para salvar as imagens
os.makedirs('p1_etapa1_regressao_linear', exist_ok=True)
os.makedirs('p1_etapa2_regressao_logistica', exist_ok=True)

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NBADataDownloader:
    """Classe para download de dados da NBA usando nba_api"""
    
    def __init__(self, season='2024-25'):
        self.season = season
        self.team_data = None
        self.player_data = None
        
    def download_team_data(self, team_name=None):
        """
        Baixa dados de times da NBA
        PRIORIDADE: Usa dados reais do CSV se disponível
        """
        # PRIMEIRO: Tentar carregar dados reais do CSV
        if team_name and team_name.lower() == 'chicago bulls' and self.season == '2024-25':
            csv_file = 'chicago_bulls_2024_25_games.csv'
            if os.path.exists(csv_file):
                print(f"✅ Carregando dados REAIS do arquivo: {csv_file}")
                try:
                    games_df = pd.read_csv(csv_file)
                    
                    # Verificar se tem dados suficientes
                    if len(games_df) >= 10:
                        print(f"✅ {len(games_df)} jogos REAIS carregados!")
                        
                        # Calcular estatísticas
                        wins = len(games_df[games_df['WL'] == 'W'])
                        losses = len(games_df[games_df['WL'] == 'L'])
                        print(f"   Record: {wins}W - {losses}L ({wins/len(games_df)*100:.1f}%)")
                        print(f"   Média de pontos: {games_df['PTS'].mean():.1f} PPG")
                        
                        # Adicionar percentuais se não existirem
                        if 'FG_PCT' not in games_df.columns and 'FGM' in games_df.columns and 'FGA' in games_df.columns:
                            games_df['FG_PCT'] = (games_df['FGM'] / games_df['FGA'] * 100).round(1)
                        if 'FG3_PCT' not in games_df.columns and 'FG3M' in games_df.columns and 'FG3A' in games_df.columns:
                            games_df['FG3_PCT'] = (games_df['FG3M'] / games_df['FG3A'] * 100).round(1)
                        if 'FT_PCT' not in games_df.columns and 'FTM' in games_df.columns and 'FTA' in games_df.columns:
                            games_df['FT_PCT'] = (games_df['FTM'] / games_df['FTA'] * 100).round(1)
                        
                        self.team_data = games_df
                        return games_df
                    else:
                        print(f"⚠️ Arquivo tem apenas {len(games_df)} jogos. Tentando API...")
                except Exception as e:
                    print(f"⚠️ Erro ao carregar CSV: {str(e)}")
                    print("   Tentando API...")
        
        # SEGUNDO: Tentar API da NBA
        try:
            from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, leaguedashteamstats
            from nba_api.stats.static import teams
            
            print(f"Baixando dados da temporada {self.season} via API...")
            
            # Obter informações dos times
            nba_teams = teams.get_teams()
            
            if team_name:
                team_dict = [team for team in nba_teams if team['full_name'].lower() == team_name.lower()]
                if not team_dict:
                    print(f"AVISO: Time '{team_name}' não encontrado. Times disponíveis:")
                    for team in nba_teams[:10]:
                        print(f"  - {team['full_name']}")
                    return None
                team_id = team_dict[0]['id']
                print(f"Time selecionado: {team_dict[0]['full_name']}")
            else:
                # Usar Lakers como padrão
                team_dict = [team for team in nba_teams if team['abbreviation'] == 'LAL']
                team_id = team_dict[0]['id']
                print(f"Time padrão selecionado: {team_dict[0]['full_name']}")
            
            # TENTATIVA 1: TeamGameLog
            print("Tentando TeamGameLog...")
            try:
                gamelog = teamgamelog.TeamGameLog(
                    team_id=team_id,
                    season=self.season,
                    season_type_all_star='Regular Season'
                )
                games_df = gamelog.get_data_frames()[0]
                
                if len(games_df) > 0:
                    print(f"✅ {len(games_df)} jogos baixados via TeamGameLog!")
                else:
                    print("⚠️ TeamGameLog retornou 0 jogos")
                    games_df = None
            except:
                games_df = None
            
            # TENTATIVA 2: LeagueGameFinder (se Tentativa 1 falhou)
            if games_df is None or len(games_df) == 0:
                print("Tentando LeagueGameFinder...")
                try:
                    gamefinder = leaguegamefinder.LeagueGameFinder(
                        team_id_nullable=team_id,
                        season_nullable=self.season,
                        season_type_nullable='Regular Season'
                    )
                    games_df = gamefinder.get_data_frames()[0]
                    
                    if len(games_df) > 0:
                        print(f"✅ {len(games_df)} jogos baixados via LeagueGameFinder!")
                    else:
                        print("⚠️ LeagueGameFinder retornou 0 jogos")
                        games_df = None
                except:
                    games_df = None
            
            # Se conseguiu baixar dados
            if games_df is not None and len(games_df) > 0:
                # Adicionar percentuais se não existirem
                if 'FG_PCT' not in games_df.columns and 'FGM' in games_df.columns and 'FGA' in games_df.columns:
                    games_df['FG_PCT'] = (games_df['FGM'] / games_df['FGA'] * 100).round(1)
                if 'FG3_PCT' not in games_df.columns and 'FG3M' in games_df.columns and 'FG3A' in games_df.columns:
                    games_df['FG3_PCT'] = (games_df['FG3M'] / games_df['FG3A'] * 100).round(1)
                if 'FT_PCT' not in games_df.columns and 'FTM' in games_df.columns and 'FTA' in games_df.columns:
                    games_df['FT_PCT'] = (games_df['FTM'] / games_df['FTA'] * 100).round(1)
                
                self.team_data = games_df
                print(f"Download completo! {len(games_df)} jogos carregados.")
                return games_df
            
            # Se API falhou, gerar dados de exemplo
            return self._generate_sample_data()
            
        except Exception as e:
            print(f"ERRO ao baixar dados: {str(e)}")
            print("Gerando dados de exemplo para demonstração...")
            return self._generate_sample_data()
    
            return games_df
            
        except Exception as e:
            print(f"ERRO ao baixar dados: {str(e)}")
            print("Gerando dados de exemplo para demonstração...")
            return self._generate_sample_data()
    
    def download_player_data(self, team_name=None):
        """
        Baixa dados de jogadores da NBA
        """
        try:
            from nba_api.stats.endpoints import playergamelog, commonteamroster
            from nba_api.stats.static import teams, players
            
            print(f"Baixando dados de jogadores da temporada {self.season}...")
            
            # Obter informações dos times
            nba_teams = teams.get_teams()
            
            if team_name:
                team_dict = [team for team in nba_teams if team['full_name'].lower() == team_name.lower()]
                if not team_dict:
                    return None
                team_id = team_dict[0]['id']
            else:
                team_dict = [team for team in nba_teams if team['abbreviation'] == 'LAL']
                team_id = team_dict[0]['id']
            
            # Obter roster do time
            print("Baixando roster do time...")
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=self.season
            )
            roster_df = roster.get_data_frames()[0]
            
            # Baixar dados de jogadores
            all_player_games = []
            
            for idx, player in roster_df.head(10).iterrows():  # Primeiros 10 jogadores
                try:
                    player_id = player['PLAYER_ID']
                    player_name = player['PLAYER']
                    
                    print(f"  Baixando dados de {player_name}...")
                    
                    player_log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=self.season,
                        season_type_all_star='Regular Season'
                    )
                    
                    player_games_df = player_log.get_data_frames()[0]
                    player_games_df['PLAYER_NAME'] = player_name
                    all_player_games.append(player_games_df)
                    
                except Exception as e:
                    print(f"  AVISO: Erro ao baixar dados de {player_name}: {str(e)}")
                    continue
            
            if all_player_games:
                self.player_data = pd.concat(all_player_games, ignore_index=True)
                print(f"Download completo! {len(self.player_data)} jogos de jogadores carregados.")
                return self.player_data
            else:
                print("AVISO: Nenhum dado de jogador foi carregado.")
                return self._generate_sample_player_data()
                
        except Exception as e:
            print(f"ERRO ao baixar dados de jogadores: {str(e)}")
            print("Gerando dados de exemplo para demonstração...")
            return self._generate_sample_player_data()
    
    def _generate_sample_data(self):
        """
        Gera dados de exemplo baseados nas estatísticas reais do Chicago Bulls 2024-25
        Dados baseados no início da temporada atual
        """
        print("Gerando dados de exemplo baseados em Chicago Bulls 2024-25...")
        
        np.random.seed(42)
        n_games = 82  # Temporada completa simulada
        
        # Estatísticas base do Chicago Bulls 2024-25 (projetadas/iniciais)
        # Baseado no histórico recente e início da temporada
        base_stats = {
            'PTS': 112.5,      # Pontos por jogo
            'FGM': 41.2,       # Arremessos convertidos
            'FGA': 88.5,       # Arremessos tentados
            'FG3M': 12.8,      # Arremessos de 3 convertidos
            'FG3A': 35.4,      # Arremessos de 3 tentados
            'FTM': 17.3,       # Lances livres convertidos
            'FTA': 21.4,       # Lances livres tentados
            'REB': 44.5,       # Rebotes
            'AST': 26.8,       # Assistências
            'STL': 7.9,        # Roubos de bola
            'BLK': 4.6,        # Tocos
            'TOV': 13.5,       # Turnovers
        }
        
        # Gerar dados com variação realista (+/- 15%)
        data = {}
        for stat, mean_value in base_stats.items():
            std_dev = mean_value * 0.12  # 12% de desvio padrão
            data[stat] = np.random.normal(mean_value, std_dev, n_games)
            
            # Garantir valores positivos e arredondar para inteiros
            if stat != 'FG_PCT' and stat != 'FG3_PCT' and stat != 'FT_PCT':
                data[stat] = np.abs(data[stat]).round(0).astype(int)
        
        # Adicionar datas (temporada 2024-25: outubro 2024 a abril 2025)
        data['GAME_DATE'] = pd.date_range(start='2024-10-22', periods=n_games, freq='2D')
        
        # Calcular percentuais de arremesso
        df = pd.DataFrame(data)
        df['FG_PCT'] = (df['FGM'] / df['FGA'] * 100).round(1)
        df['FG3_PCT'] = (df['FG3M'] / df['FG3A'] * 100).round(1)
        df['FT_PCT'] = (df['FTM'] / df['FTA'] * 100).round(1)
        
        # Calcular PLUS_MINUS baseado no desempenho
        # Melhor desempenho = PLUS_MINUS positivo
        df['PLUS_MINUS'] = (
            (df['PTS'] - 110) * 0.5 +  # Pontos acima/abaixo da média
            (df['FG_PCT'] - 46) * 0.3 +  # FG% acima/abaixo da média
            (df['AST'] - 26) * 0.4 +     # Assistências acima/abaixo
            (df['TOV'] - 13.5) * -0.5    # Menos turnovers = melhor
        ).round(0).astype(int)
        
        # Determinar vitórias/derrotas baseado em PLUS_MINUS e desempenho
        # Bulls 2024-25: ~45-50% de vitórias esperadas
        win_probability = []
        for idx, row in df.iterrows():
            # Fatores que influenciam vitória
            pts_factor = 1.0 if row['PTS'] >= 110 else 0.7
            fg_pct_factor = 1.0 if row['FG_PCT'] >= 45 else 0.8
            plus_minus_factor = 1.0 if row['PLUS_MINUS'] > 0 else 0.6
            
            # Probabilidade final
            prob = 0.47 * pts_factor * fg_pct_factor * plus_minus_factor
            win_probability.append(prob)
        
        df['WL'] = ['W' if np.random.random() < prob else 'L' 
                    for prob in win_probability]
        
        # Adicionar colunas extras para análise
        df['OREB'] = (df['REB'] * 0.28).round(0).astype(int)  # ~28% rebotes ofensivos
        df['DREB'] = df['REB'] - df['OREB']  # Rebotes defensivos
        df['PF'] = np.random.randint(18, 26, n_games)  # Faltas pessoais
        
        self.team_data = df
        
        # Estatísticas resumidas
        wins = len(df[df['WL'] == 'W'])
        losses = len(df[df['WL'] == 'L'])
        win_pct = wins / n_games * 100
        
        print(f"✅ {n_games} jogos simulados (Temporada 2024-25)")
        print(f"   Record: {wins}W - {losses}L ({win_pct:.1f}%)")
        print(f"   Média de pontos: {df['PTS'].mean():.1f} PPG")
        print(f"   FG%: {df['FG_PCT'].mean():.1f}%")
        print(f"   Rebotes/jogo: {df['REB'].mean():.1f}")
        print(f"   Assistências/jogo: {df['AST'].mean():.1f}")
        
        return df
    
    def _generate_sample_player_data(self):
        """Gera dados de exemplo de jogadores"""
        print("Gerando dados de jogadores de exemplo...")
        
        np.random.seed(42)
        players = ['LeBron James', 'Anthony Davis', 'Austin Reaves', 'D\'Angelo Russell', 'Rui Hachimura']
        n_games = 50
        
        all_data = []
        for player in players:
            data = {
                'PLAYER_NAME': [player] * n_games,
                'GAME_DATE': pd.date_range(start='2024-10-01', periods=n_games, freq='2D'),
                'PTS': np.random.randint(8, 35, n_games),
                'FGM': np.random.randint(3, 12, n_games),
                'FGA': np.random.randint(8, 22, n_games),
                'FG3M': np.random.randint(0, 5, n_games),
                'FG3A': np.random.randint(2, 10, n_games),
                'FTM': np.random.randint(2, 8, n_games),
                'FTA': np.random.randint(2, 10, n_games),
                'REB': np.random.randint(3, 12, n_games),
                'AST': np.random.randint(1, 10, n_games),
                'STL': np.random.randint(0, 3, n_games),
                'BLK': np.random.randint(0, 3, n_games),
                'TOV': np.random.randint(1, 5, n_games),
                'MIN': np.random.randint(20, 38, n_games),
                'PLUS_MINUS': np.random.randint(-12, 15, n_games)
            }
            all_data.append(pd.DataFrame(data))
        
        df = pd.concat(all_data, ignore_index=True)
        df['FG_PCT'] = (df['FGM'] / df['FGA'] * 100).round(1)
        df['FG3_PCT'] = (df['FG3M'] / df['FG3A'] * 100).round(1)
        df['FT_PCT'] = (df['FTM'] / df['FTA'] * 100).round(1)
        
        self.player_data = df
        print(f"{len(df)} jogos de jogadores de exemplo gerados.")
        return df


class LinearRegressionAnalysis:
    """Classe para análise de Regressão Linear"""
    
    def __init__(self, data):
        self.data = data
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.target_variable = None
        
    def prepare_data(self, target_variable, independent_variables):
        """
        Prepara os dados para a regressão
        
        Args:
            target_variable (str): Variável dependente (Y)
            independent_variables (list): Lista de variáveis independentes (X)
        """
        print(f"\nPreparando dados para análise...")
        print(f"   Variável dependente (Y): {target_variable}")
        print(f"   Variáveis independentes (X): {', '.join(independent_variables)}")
        
        self.target_variable = target_variable
        self.selected_features = independent_variables
        
        # Remover valores nulos
        data_clean = self.data[independent_variables + [target_variable]].dropna()
        
        # Remover valores infinitos
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"   {len(data_clean)} amostras válidas após limpeza")
        
        if len(data_clean) < 10:
            raise ValueError("ERRO: Dados insuficientes para análise. Mínimo: 10 amostras.")
        
        # Separar variáveis
        X = data_clean[independent_variables].values
        y = data_clean[target_variable].values
        
        # Dividir em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Treino: {len(self.X_train)} amostras | Teste: {len(self.X_test)} amostras")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Treina o modelo de regressão linear"""
        print("\nTreinando modelo de Regressão Linear...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Fazer previsões
        self.y_pred = self.model.predict(self.X_test)
        
        print("   Modelo treinado com sucesso!")
        
        return self.model
    
    def evaluate_model(self):
        """Avalia o desempenho do modelo"""
        print("\nAvaliação do Modelo:")
        print("=" * 60)
        
        # Métricas
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE (Erro Quadrático Médio): {mse:.4f}")
        print(f"   RMSE (Raiz do MSE): {rmse:.4f}")
        print(f"   MAE (Erro Absoluto Médio): {mae:.4f}")
        
        # Equação da regressão
        print("\nEquação da Regressão Linear:")
        print("=" * 60)
        print(f"   y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε")
        print(f"\n   y ({self.target_variable}) = {self.model.intercept_:.4f}", end="")
        
        for i, (feature, coef) in enumerate(zip(self.selected_features, self.model.coef_)):
            sign = "+" if coef >= 0 else ""
            print(f" {sign} {coef:.4f}×{feature}", end="")
        
        print("\n")
        
        # Coeficientes
        print("Coeficientes (β) e Impacto das Variáveis:")
        print("=" * 60)
        print(f"   β₀ (Intercepto): {self.model.intercept_:.4f}")
        
        for i, (feature, coef) in enumerate(zip(self.selected_features, self.model.coef_), 1):
            impact = "positivo" if coef > 0 else "negativo"
            print(f"   β{i} ({feature}): {coef:.4f} (impacto {impact})")
        
        print("=" * 60)
        
        metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.selected_features, self.model.coef_))
        }
        
        return metrics
    
    def predict(self, X_new):
        """Faz previsões com novos dados"""
        if self.model is None:
            raise ValueError("ERRO: Modelo não foi treinado ainda!")
        
        return self.model.predict(X_new)
    
    def plot_scatter_with_regression(self):
        """Diagrama de Dispersão com Linha de Regressão"""
        print("\nGerando Diagrama de Dispersão com Linha de Regressão...")
        
        if len(self.selected_features) == 1:
            # Regressão simples - plotar linha diretamente
            plt.figure(figsize=(12, 6))
            
            # Dados de treino
            plt.subplot(1, 2, 1)
            plt.scatter(self.X_train, self.y_train, alpha=0.6, label='Dados de Treino')
            
            # Linha de regressão
            X_line = np.linspace(self.X_train.min(), self.X_train.max(), 100).reshape(-1, 1)
            y_line = self.model.predict(X_line)
            plt.plot(X_line, y_line, 'r-', linewidth=2, label='Linha de Regressão')
            
            plt.xlabel(self.selected_features[0], fontsize=12)
            plt.ylabel(self.target_variable, fontsize=12)
            plt.title('Dados de Treino', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Dados de teste
            plt.subplot(1, 2, 2)
            plt.scatter(self.X_test, self.y_test, alpha=0.6, color='green', label='Dados de Teste')
            plt.scatter(self.X_test, self.y_pred, alpha=0.6, color='red', label='Previsões')
            
            plt.xlabel(self.selected_features[0], fontsize=12)
            plt.ylabel(self.target_variable, fontsize=12)
            plt.title('Dados de Teste vs Previsões', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            # Regressão múltipla - usar gráfico de valores reais vs preditos
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot para cada variável independente
            for idx, feature in enumerate(self.selected_features[:2]):  # Primeiras 2 variáveis
                ax = axes[idx] if len(self.selected_features) > 1 else axes
                
                feature_idx = self.selected_features.index(feature)
                ax.scatter(self.X_test[:, feature_idx], self.y_test, alpha=0.6, label='Valores Reais')
                ax.scatter(self.X_test[:, feature_idx], self.y_pred, alpha=0.6, color='red', label='Previsões')
                
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel(self.target_variable, fontsize=12)
                ax.set_title(f'{self.target_variable} vs {feature}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('p1_etapa1_regressao_linear/scatter_regression.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa1_regressao_linear/scatter_regression.png")
        plt.show()
    
    def plot_prediction_vs_reality(self):
        """Gráfico de Previsão vs. Realidade"""
        print("\nGerando Gráfico de Previsão vs. Realidade...")
        
        plt.figure(figsize=(10, 8))
        
        # Plot principal
        plt.scatter(self.y_test, self.y_pred, alpha=0.6, s=80)
        
        # Linha ideal (y = x)
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
        
        # Métricas no gráfico
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.xlabel('Valores Reais', fontsize=12)
        plt.ylabel('Valores Previstos', fontsize=12)
        plt.title('Previsão vs. Realidade', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('p1_etapa1_regressao_linear/prediction_vs_reality.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa1_regressao_linear/prediction_vs_reality.png")
        plt.show()
    
    def plot_confusion_matrix_regression(self):
        """Gráfico de Matriz de Confusão para Regressão (binned)"""
        print("\nGerando Matriz de Confusão (Regressão Binned)...")
        
        # Criar bins para categorizar valores
        n_bins = 5
        y_test_binned = pd.cut(self.y_test, bins=n_bins, labels=[f'Bin {i+1}' for i in range(n_bins)])
        y_pred_binned = pd.cut(self.y_pred, bins=n_bins, labels=[f'Bin {i+1}' for i in range(n_bins)])
        
        # Criar matriz de confusão
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_binned, y_pred_binned)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Bin {i+1}' for i in range(n_bins)],
                   yticklabels=[f'Bin {i+1}' for i in range(n_bins)])
        
        plt.xlabel('Previsões (Bins)', fontsize=12)
        plt.ylabel('Valores Reais (Bins)', fontsize=12)
        plt.title(f'Matriz de Confusão - {self.target_variable}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('p1_etapa1_regressao_linear/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa1_regressao_linear/confusion_matrix.png")
        plt.show()
    
    def plot_trend_with_confidence(self):
        """Gráfico de Tendência com Intervalo de Confiança"""
        print("\nGerando Gráfico de Tendência com Intervalo de Confiança...")
        
        # Ordenar dados para melhor visualização
        sorted_indices = np.argsort(self.y_test)
        y_test_sorted = self.y_test[sorted_indices]
        y_pred_sorted = self.y_pred[sorted_indices]
        
        # Calcular intervalo de confiança (95%)
        residuals = y_test_sorted - y_pred_sorted
        std_residuals = np.std(residuals)
        confidence_interval = 1.96 * std_residuals
        
        plt.figure(figsize=(12, 6))
        
        # Plot da tendência
        x_range = np.arange(len(y_test_sorted))
        plt.plot(x_range, y_test_sorted, 'o-', label='Valores Reais', markersize=6, alpha=0.7)
        plt.plot(x_range, y_pred_sorted, 's-', label='Valores Previstos', markersize=6, alpha=0.7)
        
        # Intervalo de confiança
        plt.fill_between(x_range, 
                        y_pred_sorted - confidence_interval,
                        y_pred_sorted + confidence_interval,
                        alpha=0.2, label='Intervalo de Confiança (95%)')
        
        plt.xlabel('Índice (ordenado)', fontsize=12)
        plt.ylabel(self.target_variable, fontsize=12)
        plt.title(f'Tendência com Intervalo de Confiança - {self.target_variable}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('p1_etapa1_regressao_linear/trend_confidence.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa1_regressao_linear/trend_confidence.png")
        plt.show()
    
    def generate_all_plots(self):
        """Gera todos os gráficos de uma vez"""
        print("\nGerando todos os gráficos...")
        
        self.plot_scatter_with_regression()
        self.plot_prediction_vs_reality()
        self.plot_confusion_matrix_regression()
        self.plot_trend_with_confidence()
        
        print("\nTodos os gráficos foram gerados e salvos!")


class LogisticRegressionAnalysis:
    """Classe para análise de Regressão Logística"""
    
    def __init__(self, data):
        self.data = data
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.target_variable = None
        
    def prepare_data(self, target_variable, independent_variables):
        """
        Prepara os dados para a regressão logística
        
        Args:
            target_variable (str): Variável dependente binária (Y) - ex: WL (Vitória/Derrota)
            independent_variables (list): Lista de variáveis independentes (X)
        """
        print(f"\nPreparando dados para Regressão Logística...")
        print(f"   Variável dependente (Y): {target_variable}")
        print(f"   Variáveis independentes (X): {', '.join(independent_variables)}")
        
        self.target_variable = target_variable
        self.selected_features = independent_variables
        
        # Remover valores nulos
        data_clean = self.data[independent_variables + [target_variable]].dropna()
        
        # Remover valores infinitos
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"   {len(data_clean)} amostras válidas após limpeza")
        
        if len(data_clean) < 10:
            raise ValueError("ERRO: Dados insuficientes para análise. Mínimo: 10 amostras.")
        
        # Separar variáveis
        X = data_clean[independent_variables].values
        y_raw = data_clean[target_variable].values
        
        # Converter Y para binário (0 ou 1)
        if y_raw.dtype == 'object' or isinstance(y_raw[0], str):
            # Se for categórico (ex: 'W' ou 'L')
            unique_values = np.unique(y_raw)
            print(f"   Convertendo variável categórica: {unique_values}")
            
            # Mapear para binário
            if len(unique_values) == 2:
                y = (y_raw == unique_values[0]).astype(int)
                print(f"      {unique_values[0]} = 1 (Vitória)")
                print(f"      {unique_values[1]} = 0 (Derrota)")
            else:
                raise ValueError(f"ERRO: Variável {target_variable} deve ser binária!")
        else:
            y = y_raw.astype(int)
        
        # Dividir em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Treino: {len(self.X_train)} amostras | Teste: {len(self.X_test)} amostras")
        print(f"   Distribuição treino - Vitórias: {np.sum(self.y_train)}, Derrotas: {len(self.y_train) - np.sum(self.y_train)}")
        print(f"   Distribuição teste - Vitórias: {np.sum(self.y_test)}, Derrotas: {len(self.y_test) - np.sum(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Treina o modelo de regressão logística"""
        print("\nTreinando modelo de Regressão Logística...")
        
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Fazer previsões
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  # Probabilidade da classe 1
        
        print("   Modelo treinado com sucesso!")
        
        return self.model
    
    def evaluate_model(self):
        """Avalia o desempenho do modelo"""
        print("\nAvaliação do Modelo de Regressão Logística:")
        print("=" * 70)
        
        # Métricas
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(self.y_test, self.y_pred_proba)
        except:
            auc = 0.0
        
        print(f"   Acurácia (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precisão (Precision): {precision:.4f}")
        print(f"   Recall (Sensibilidade): {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC-ROC: {auc:.4f}")
        
        # Equação da regressão logística
        print("\nEquação da Regressão Logística:")
        print("=" * 70)
        print(f"   p = 1 / [1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))]")
        print(f"\n   Onde p é a probabilidade de {self.target_variable} = 1 (Vitória)")
        print(f"\n   z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
        print(f"   z = {self.model.intercept_[0]:.4f}", end="")
        
        for i, (feature, coef) in enumerate(zip(self.selected_features, self.model.coef_[0])):
            sign = "+" if coef >= 0 else ""
            print(f" {sign} {coef:.4f}×{feature}", end="")
        
        print("\n")
        
        # Coeficientes e odds ratios
        print("Coeficientes (β) e Impacto das Variáveis:")
        print("=" * 70)
        print(f"   β₀ (Intercepto): {self.model.intercept_[0]:.4f}")
        
        for i, (feature, coef) in enumerate(zip(self.selected_features, self.model.coef_[0]), 1):
            odds_ratio = np.exp(coef)
            impact = "aumenta" if coef > 0 else "diminui"
            print(f"   β{i} ({feature}): {coef:.4f}")
            print(f"      → Odds Ratio: {odds_ratio:.4f} ({impact} a chance de vitória)")
        
        print("=" * 70)
        
        # Relatório de classificação
        print("\nRelatório de Classificação:")
        print("=" * 70)
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['Derrota (0)', 'Vitória (1)'],
                                   zero_division=0))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'intercept': self.model.intercept_[0],
            'coefficients': dict(zip(self.selected_features, self.model.coef_[0]))
        }
        
        return metrics
    
    def predict_probability(self, X_new, team_name="Time", opponent_name="Oponente"):
        """
        Faz previsões de probabilidade com novos dados
        
        Args:
            X_new: Array com as features do jogo
            team_name: Nome do time
            opponent_name: Nome do oponente
        """
        if self.model is None:
            raise ValueError("ERRO: Modelo não foi treinado ainda!")
        
        proba = self.model.predict_proba(X_new)[0]
        prediction = self.model.predict(X_new)[0]
        
        win_proba = proba[1] * 100
        loss_proba = proba[0] * 100
        
        print("\n" + "=" * 70)
        print("PREVISÃO DE PROBABILIDADE DE VITÓRIA")
        print("=" * 70)
        print(f"\n   {team_name} vs {opponent_name}")
        print(f"\n   Probabilidade de VITÓRIA: {win_proba:.2f}%")
        print(f"   Probabilidade de DERROTA: {loss_proba:.2f}%")
        
        if prediction == 1:
            print(f"\n   PREVISÃO: {team_name} tem {win_proba:.0f}% de chance de VENCER!")
        else:
            print(f"\n   PREVISÃO: {team_name} tem {loss_proba:.0f}% de chance de PERDER.")
        
        print("=" * 70)
        
        return proba, prediction
    
    def plot_scatter_with_sigmoid(self):
        """Diagrama de Dispersão com Curva Sigmoide"""
        print("\nGerando Diagrama de Dispersão com Curva Sigmoide...")
        
        if len(self.selected_features) >= 1:
            fig, axes = plt.subplots(1, min(2, len(self.selected_features)), figsize=(14, 6))
            
            if len(self.selected_features) == 1:
                axes = [axes]
            
            for idx, feature in enumerate(self.selected_features[:2]):
                ax = axes[idx] if len(self.selected_features) > 1 else axes[0]
                
                feature_idx = self.selected_features.index(feature)
                
                # Scatter plot
                ax.scatter(self.X_test[:, feature_idx][self.y_test == 0], 
                          self.y_test[self.y_test == 0],
                          alpha=0.6, label='Derrota', color='red', s=80)
                ax.scatter(self.X_test[:, feature_idx][self.y_test == 1], 
                          self.y_test[self.y_test == 1],
                          alpha=0.6, label='Vitória', color='green', s=80)
                
                # Curva sigmoide
                X_range = np.linspace(self.X_test[:, feature_idx].min(), 
                                     self.X_test[:, feature_idx].max(), 300)
                
                # Para plotar a sigmoide, usar apenas essa feature
                X_sigmoid = np.zeros((300, len(self.selected_features)))
                X_sigmoid[:, feature_idx] = X_range
                
                # Usar valores médios para outras features
                for i in range(len(self.selected_features)):
                    if i != feature_idx:
                        X_sigmoid[:, i] = np.mean(self.X_train[:, i])
                
                y_sigmoid = self.model.predict_proba(X_sigmoid)[:, 1]
                ax.plot(X_range, y_sigmoid, 'b-', linewidth=3, label='Curva Sigmoide', alpha=0.7)
                
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Probabilidade de Vitória', fontsize=12)
                ax.set_title(f'Regressão Logística: {feature}', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.1, 1.1])
            
            plt.tight_layout()
            plt.savefig('p1_etapa2_regressao_logistica/logistic_scatter_sigmoid.png', dpi=300, bbox_inches='tight')
            print("   Gráfico salvo: p1_etapa2_regressao_logistica/logistic_scatter_sigmoid.png")
            plt.show()
    
    def plot_roc_curve(self):
        """Curva ROC (Receiver Operating Characteristic)"""
        print("\nGerando Curva ROC...")
        
        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
            auc = roc_auc_score(self.y_test, self.y_pred_proba)
            
            plt.figure(figsize=(10, 8))
            
            # Curva ROC
            plt.plot(fpr, tpr, linewidth=3, label=f'Curva ROC (AUC = {auc:.4f})', color='darkorange')
            
            # Linha diagonal (classificador aleatório)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Classificador Aleatório (AUC = 0.5)')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
            plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
            plt.title('Curva ROC - Regressão Logística', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Adicionar texto informativo
            textstr = f'AUC = {auc:.4f}\n'
            textstr += f'Interpretação:\n'
            if auc >= 0.9:
                textstr += 'Excelente discriminação'
            elif auc >= 0.8:
                textstr += 'Boa discriminação'
            elif auc >= 0.7:
                textstr += 'Discriminação aceitável'
            else:
                textstr += 'Discriminação fraca'
            
            plt.text(0.6, 0.2, textstr, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('p1_etapa2_regressao_logistica/roc_curve.png', dpi=300, bbox_inches='tight')
            print("   Gráfico salvo: p1_etapa2_regressao_logistica/roc_curve.png")
            plt.show()
            
        except Exception as e:
            print(f"   AVISO: Erro ao gerar curva ROC: {str(e)}")
    
    def plot_predicted_probabilities(self):
        """Gráfico de Probabilidades Previstas"""
        print("\nGerando Gráfico de Probabilidades Previstas...")
        
        plt.figure(figsize=(12, 6))
        
        # Separar probabilidades por classe real
        proba_victory = self.y_pred_proba[self.y_test == 1]
        proba_defeat = self.y_pred_proba[self.y_test == 0]
        
        # Histogramas
        plt.hist(proba_defeat, bins=20, alpha=0.6, label='Derrotas Reais', color='red', edgecolor='black')
        plt.hist(proba_victory, bins=20, alpha=0.6, label='Vitórias Reais', color='green', edgecolor='black')
        
        # Linha de threshold (0.5)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        plt.xlabel('Probabilidade Prevista de Vitória', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição de Probabilidades Previstas', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('p1_etapa2_regressao_logistica/predicted_probabilities.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa2_regressao_logistica/predicted_probabilities.png")
        plt.show()
    
    def plot_feature_importance(self):
        """Gráfico de Importância de Variáveis"""
        print("\nGerando Gráfico de Importância de Variáveis...")
        
        # Coeficientes do modelo (magnitude absoluta representa importância)
        coefficients = self.model.coef_[0]
        feature_importance = np.abs(coefficients)
        
        # Criar DataFrame para ordenação
        importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': feature_importance,
            'Coefficient': coefficients
        }).sort_values('Importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, max(6, len(self.selected_features) * 0.5)))
        
        colors = ['green' if x > 0 else 'red' for x in importance_df['Coefficient']]
        
        plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.7, edgecolor='black')
        
        plt.xlabel('Importância (|Coeficiente|)', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)
        plt.title('Importância das Variáveis na Previsão de Vitória', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Impacto Positivo'),
                          Patch(facecolor='red', alpha=0.7, label='Impacto Negativo')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('p1_etapa2_regressao_logistica/feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa2_regressao_logistica/feature_importance.png")
        plt.show()
    
    def plot_confusion_matrix(self):
        """Gráfico de Matriz de Confusão"""
        print("\nGerando Matriz de Confusão...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Derrota (0)', 'Vitória (1)'],
                   yticklabels=['Derrota (0)', 'Vitória (1)'],
                   annot_kws={"size": 16})
        
        plt.xlabel('Previsão', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.title('Matriz de Confusão - Regressão Logística', fontsize=14, fontweight='bold')
        
        # Adicionar percentagens
        total = np.sum(cm)
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig('p1_etapa2_regressao_logistica/logistic_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa2_regressao_logistica/logistic_confusion_matrix.png")
        plt.show()
    
    def plot_trend_with_confidence(self):
        """Gráfico de Tendência de Probabilidades com Intervalo de Confiança"""
        print("\nGerando Gráfico de Tendência com Intervalo de Confiança...")
        
        # Ordenar por probabilidade prevista
        sorted_indices = np.argsort(self.y_pred_proba)
        y_proba_sorted = self.y_pred_proba[sorted_indices]
        y_test_sorted = self.y_test[sorted_indices]
        
        # Calcular média móvel e intervalo de confiança
        window = max(5, len(y_proba_sorted) // 10)
        
        moving_avg = pd.Series(y_proba_sorted).rolling(window=window, center=True).mean()
        moving_std = pd.Series(y_proba_sorted).rolling(window=window, center=True).std()
        
        # Intervalo de confiança (95%)
        confidence_interval = 1.96 * moving_std
        
        plt.figure(figsize=(14, 6))
        
        # Valores reais
        colors = ['red' if y == 0 else 'green' for y in y_test_sorted]
        plt.scatter(range(len(y_test_sorted)), y_test_sorted, c=colors, alpha=0.5, s=50, label='Valores Reais')
        
        # Probabilidades previstas
        plt.plot(range(len(y_proba_sorted)), y_proba_sorted, 'b-', alpha=0.6, linewidth=2, label='Probabilidade Prevista')
        
        # Média móvel
        plt.plot(range(len(moving_avg)), moving_avg, 'darkblue', linewidth=3, label=f'Média Móvel (janela={window})')
        
        # Intervalo de confiança
        plt.fill_between(range(len(moving_avg)),
                        moving_avg - confidence_interval,
                        moving_avg + confidence_interval,
                        alpha=0.2, color='blue', label='Intervalo de Confiança (95%)')
        
        # Linha de threshold
        plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        plt.xlabel('Índice (ordenado por probabilidade)', fontsize=12)
        plt.ylabel('Probabilidade / Resultado', fontsize=12)
        plt.title('Tendência de Probabilidades com Intervalo de Confiança', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim([-0.1, 1.1])
        
        plt.tight_layout()
        plt.savefig('p1_etapa2_regressao_logistica/logistic_trend_confidence.png', dpi=300, bbox_inches='tight')
        print("   Gráfico salvo: p1_etapa2_regressao_logistica/logistic_trend_confidence.png")
        plt.show()
    
    def generate_all_plots(self):
        """Gera todos os gráficos de uma vez"""
        print("\nGerando todos os gráficos de Regressão Logística...")
        
        self.plot_scatter_with_sigmoid()
        self.plot_roc_curve()
        self.plot_predicted_probabilities()
        self.plot_feature_importance()
        self.plot_confusion_matrix()
        self.plot_trend_with_confidence()
        
        print("\nTodos os gráficos de Regressão Logística foram gerados e salvos!")


def display_menu():
    """Exibe o menu principal"""
    print("\n" + "=" * 70)
    print("SISTEMA DE ANÁLISE DE REGRESSÃO - NBA 2024-25")
    print("=" * 70)
    print("\n1. Baixar dados de TIMES")
    print("2. Baixar dados de JOGADORES")
    print("3. Visualizar dados carregados")
    print("4. Executar Regressão LINEAR")
    print("5. Gerar gráficos da Regressão Linear")
    print("6. Executar Regressão LOGÍSTICA")
    print("7. Gerar gráficos da Regressão Logística")
    print("8. Fazer previsão de probabilidade de vitória")
    print("9. Sair")
    print("\n" + "=" * 70)


def display_variables(data):
    """Exibe as variáveis disponíveis nos dados"""
    print("\nVariáveis disponíveis:")
    print("=" * 70)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for i, col in enumerate(numeric_cols, 1):
        print(f"   {i:2d}. {col}")
    
    print("=" * 70)
    return numeric_cols


def select_variables(numeric_cols):
    """Permite ao usuário selecionar variáveis"""
    print("\nSeleção de Variáveis:")
    print("-" * 70)
    
    # Selecionar variável dependente (Y)
    print("\nSelecione a variável DEPENDENTE (Y) - o que você quer prever:")
    print("   Exemplos: PTS (pontos), REB (rebotes), AST (assistências)")
    
    y_idx = int(input("\n   Digite o número da variável Y: ")) - 1
    target_variable = numeric_cols[y_idx]
    
    print(f"\n   Variável Y selecionada: {target_variable}")
    
    # Selecionar variáveis independentes (X)
    print("\nSelecione as variáveis INDEPENDENTES (X) - features para previsão:")
    print("   Digite os números separados por vírgula (ex: 1,3,5,7)")
    print("   Exemplos: FGM, FGA, FG3M, AST, REB, etc.")
    
    x_input = input("\n   Digite os números das variáveis X: ")
    x_indices = [int(x.strip()) - 1 for x in x_input.split(',')]
    independent_variables = [numeric_cols[i] for i in x_indices]
    
    print(f"\n   Variáveis X selecionadas: {', '.join(independent_variables)}")
    
    return target_variable, independent_variables


def main():
    """Função principal"""
    downloader = NBADataDownloader(season='2024-25')
    data = None
    linear_analysis = None
    logistic_analysis = None
    
    print("\n" + "=" * 70)
    print("BEM-VINDO AO SISTEMA DE ANÁLISE DE REGRESSÃO NBA")
    print("=" * 70)
    print("\nEste sistema permite:")
    print("   • Baixar dados da NBA usando nba_api")
    print("   • Realizar análises de Regressão Linear")
    print("   • Realizar análises de Regressão Logística")
    print("   • Escolher variáveis dependentes e independentes")
    print("   • Gerar gráficos de análise")
    print("   • Prever probabilidades de vitória")
    print("   • Prever resultados de jogos e jogadores")
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEscolha uma opção: ").strip()
            
            if choice == '1':
                # Baixar dados de times
                print("\n" + "=" * 70)
                print("DOWNLOAD DE DADOS DE TIMES")
                print("=" * 70)
                
                team_name = input("\n   Digite o nome do time (ou deixe em branco para Lakers): ").strip()
                
                if not team_name:
                    team_name = None
                
                data = downloader.download_team_data(team_name)
                
                if data is not None:
                    print(f"\nDados carregados com sucesso!")
                    print(f"   {len(data)} jogos disponíveis")
                    print(f"   {len(data.columns)} variáveis disponíveis")
            
            elif choice == '2':
                # Baixar dados de jogadores
                print("\n" + "=" * 70)
                print("DOWNLOAD DE DADOS DE JOGADORES")
                print("=" * 70)
                
                team_name = input("\n   Digite o nome do time (ou deixe em branco para Lakers): ").strip()
                
                if not team_name:
                    team_name = None
                
                data = downloader.download_player_data(team_name)
                
                if data is not None:
                    print(f"\nDados carregados com sucesso!")
                    print(f"   {len(data)} jogos de jogadores disponíveis")
                    print(f"   {len(data.columns)} variáveis disponíveis")
            
            elif choice == '3':
                # Visualizar dados
                if data is None:
                    print("\nAVISO: Nenhum dado foi carregado ainda! Use as opções 1 ou 2 primeiro.")
                    continue
                
                print("\n" + "=" * 70)
                print("VISUALIZAÇÃO DOS DADOS")
                print("=" * 70)
                
                print("\nPrimeiras 10 linhas dos dados:")
                print(data.head(10).to_string())
                
                print("\nEstatísticas descritivas:")
                print(data.describe().to_string())
                
                display_variables(data)
            
            elif choice == '4':
                # Executar Regressão Linear
                if data is None:
                    print("\nAVISO: Nenhum dado foi carregado ainda! Use as opções 1 ou 2 primeiro.")
                    continue
                
                print("\n" + "=" * 70)
                print("ANÁLISE DE REGRESSÃO LINEAR")
                print("=" * 70)
                
                # Mostrar variáveis disponíveis
                numeric_cols = display_variables(data)
                
                # Selecionar variáveis
                target_variable, independent_variables = select_variables(numeric_cols)
                
                # Criar análise
                linear_analysis = LinearRegressionAnalysis(data)
                
                # Preparar dados
                try:
                    linear_analysis.prepare_data(target_variable, independent_variables)
                    
                    # Treinar modelo
                    linear_analysis.train_model()
                    
                    # Avaliar modelo
                    metrics = linear_analysis.evaluate_model()
                    
                    print("\nAnálise de Regressão Linear concluída!")
                    
                    # Perguntar se deseja gerar gráficos
                    gen_plots = input("\nDeseja gerar os gráficos agora? (s/n): ").strip().lower()
                    if gen_plots == 's':
                        linear_analysis.generate_all_plots()
                    
                except Exception as e:
                    print(f"\nERRO na análise: {str(e)}")
            
            elif choice == '5':
                # Gerar gráficos da Regressão Linear
                if linear_analysis is None:
                    print("\nAVISO: Execute a análise de regressão linear primeiro (opção 4)!")
                    continue
                
                linear_analysis.generate_all_plots()
            
            elif choice == '6':
                # Executar Regressão Logística
                if data is None:
                    print("\nAVISO: Nenhum dado foi carregado ainda! Use as opções 1 ou 2 primeiro.")
                    continue
                
                print("\n" + "=" * 70)
                print("ANÁLISE DE REGRESSÃO LOGÍSTICA")
                print("=" * 70)
                print("\nA Regressão Logística prevê probabilidades de vitória/derrota")
                print("   usando a função sigmoide: p = 1 / [1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ)]")
                
                # Verificar se existe variável binária
                # Procurar coluna WL (Win/Loss) ou criar uma
                if 'WL' not in data.columns:
                    print("\nAVISO: Coluna 'WL' (Vitória/Derrota) não encontrada.")
                    print("   Criando variável binária baseada em PLUS_MINUS...")
                    
                    if 'PLUS_MINUS' in data.columns:
                        data['WL'] = (data['PLUS_MINUS'] > 0).astype(int)
                        data['WL'] = data['WL'].map({1: 'W', 0: 'L'})
                        print("   Variável 'WL' criada: W (vitória) se PLUS_MINUS > 0")
                    else:
                        print("   ERRO: Não foi possível criar variável binária.")
                        print("   Certifique-se de que os dados contêm uma coluna 'WL' ou 'PLUS_MINUS'")
                        continue
                
                # Mostrar variáveis disponíveis
                numeric_cols = display_variables(data)
                
                print("\nSelecione a variável DEPENDENTE (Y) - deve ser binária (WL):")
                print("   IMPORTANTE: Use 'WL' para prever Vitória/Derrota")
                
                # Para regressão logística, a variável Y deve ser WL
                if 'WL' in data.columns:
                    target_variable = 'WL'
                    print(f"\n   Variável Y: {target_variable} (Vitória/Derrota)")
                else:
                    print("\n   ERRO: Variável 'WL' não disponível!")
                    continue
                
                # Selecionar variáveis independentes
                print("\nSelecione as variáveis INDEPENDENTES (X):")
                print("   Digite os números separados por vírgula (ex: 1,3,5,7)")
                print("   Sugestões: PTS, FGM, FGA, FG3M, REB, AST, STL, BLK, etc.")
                
                x_input = input("\n   Digite os números das variáveis X: ")
                x_indices = [int(x.strip()) - 1 for x in x_input.split(',')]
                independent_variables = [numeric_cols[i] for i in x_indices]
                
                print(f"\n   Variáveis X selecionadas: {', '.join(independent_variables)}")
                
                # Criar análise
                logistic_analysis = LogisticRegressionAnalysis(data)
                
                # Preparar dados
                try:
                    logistic_analysis.prepare_data(target_variable, independent_variables)
                    
                    # Treinar modelo
                    logistic_analysis.train_model()
                    
                    # Avaliar modelo
                    metrics = logistic_analysis.evaluate_model()
                    
                    print("\nAnálise de Regressão Logística concluída!")
                    
                    # Perguntar se deseja gerar gráficos
                    gen_plots = input("\nDeseja gerar os gráficos agora? (s/n): ").strip().lower()
                    if gen_plots == 's':
                        logistic_analysis.generate_all_plots()
                    
                except Exception as e:
                    print(f"\nERRO na análise: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '7':
                # Gerar gráficos da Regressão Logística
                if logistic_analysis is None:
                    print("\nAVISO: Execute a análise de regressão logística primeiro (opção 6)!")
                    continue
                
                logistic_analysis.generate_all_plots()
            
            elif choice == '8':
                # Fazer previsão de probabilidade
                if logistic_analysis is None:
                    print("\nAVISO: Execute a análise de regressão logística primeiro (opção 6)!")
                    continue
                
                print("\n" + "=" * 70)
                print("PREVISÃO DE PROBABILIDADE DE VITÓRIA")
                print("=" * 70)
                
                print("\nInsira os valores das variáveis para fazer a previsão:")
                print(f"   Variáveis necessárias: {', '.join(logistic_analysis.selected_features)}")
                
                try:
                    X_new = []
                    for feature in logistic_analysis.selected_features:
                        value = float(input(f"\n   {feature}: "))
                        X_new.append(value)
                    
                    X_new = np.array([X_new])
                    
                    # Perguntar nomes dos times
                    team_name = input("\n   Nome do seu time: ").strip()
                    if not team_name:
                        team_name = "Seu Time"
                    
                    opponent_name = input("   Nome do oponente: ").strip()
                    if not opponent_name:
                        opponent_name = "Oponente"
                    
                    # Fazer previsão
                    logistic_analysis.predict_probability(X_new, team_name, opponent_name)
                    
                except Exception as e:
                    print(f"\nERRO na previsão: {str(e)}")
            
            elif choice == '9':
                # Sair
                print("\n" + "=" * 70)
                print("Obrigado por usar o Sistema de Análise NBA!")
                print("=" * 70)
                break
            
            else:
                print("\nAVISO: Opção inválida! Tente novamente.")
        
        except KeyboardInterrupt:
            print("\n\nAVISO: Operação cancelada pelo usuário.")
            break
        except Exception as e:
            print(f"\nERRO: {str(e)}")
            print("Tente novamente ou escolha outra opção.")


def executar_analise_completa_bulls():
    """
    Executa análise completa automatizada para Chicago Bulls - Temporada 2024-25
    USA APENAS DADOS REAIS - Sem simulação
    """
    
    print("\n" + "=" * 80)
    print("ANÁLISE COMPLETA - CHICAGO BULLS 2024-25 (DADOS REAIS)")
    print("=" * 80)
    
    # ETAPA 1: Carregar dados REAIS
    print("\n[1/7] Carregando dados REAIS do Chicago Bulls (Temporada 2024-25)...")
    
    # Verificar se arquivo CSV existe
    csv_file = 'chicago_bulls_2024_25_games.csv'
    if not os.path.exists(csv_file):
        print(f"\n❌ ERRO: Arquivo '{csv_file}' não encontrado!")
        print("\n📥 SOLUÇÃO: Execute primeiro o script de download:")
        print("   python download_bulls_data.py")
        print("\nEste script irá:")
        print("   1. Conectar na API da NBA")
        print("   2. Baixar todos os jogos do Chicago Bulls 2024-25")
        print("   3. Salvar em 'chicago_bulls_2024_25_games.csv'")
        print("\nDepois execute novamente este programa.")
        return
    
    # Carregar dados reais do CSV
    try:
        data = pd.read_csv(csv_file)
        print(f"✅ Arquivo carregado: {csv_file}")
        print(f"✅ {len(data)} jogos REAIS encontrados!")
        
        # Validar dados mínimos
        if len(data) < 10:
            print(f"\n❌ ERRO: Apenas {len(data)} jogos encontrados!")
            print("   São necessários no mínimo 10 jogos para análise confiável.")
            print("\n💡 Execute novamente 'python download_bulls_data.py'")
            return
        
        # Calcular e exibir estatísticas
        wins = len(data[data['WL'] == 'W'])
        losses = len(data[data['WL'] == 'L'])
        win_pct = wins / len(data) * 100
        
        print(f"\n📊 ESTATÍSTICAS DA TEMPORADA 2024-25:")
        print(f"   Total de jogos: {len(data)}")
        print(f"   Record: {wins}W - {losses}L ({win_pct:.1f}%)")
        print(f"   Média de pontos: {data['PTS'].mean():.1f} PPG")
        
        if 'FG_PCT' in data.columns:
            print(f"   Field Goal %: {data['FG_PCT'].mean():.1f}%")
        if 'FG3_PCT' in data.columns:
            print(f"   3-Point %: {data['FG3_PCT'].mean():.1f}%")
        if 'REB' in data.columns:
            print(f"   Rebotes/jogo: {data['REB'].mean():.1f}")
        if 'AST' in data.columns:
            print(f"   Assistências/jogo: {data['AST'].mean():.1f}")
        
        # Adicionar percentuais se não existirem
        if 'FG_PCT' not in data.columns and 'FGM' in data.columns and 'FGA' in data.columns:
            data['FG_PCT'] = (data['FGM'] / data['FGA'] * 100).round(1)
        if 'FG3_PCT' not in data.columns and 'FG3M' in data.columns and 'FG3A' in data.columns:
            data['FG3_PCT'] = (data['FG3M'] / data['FG3A'] * 100).round(1)
        if 'FT_PCT' not in data.columns and 'FTM' in data.columns and 'FTA' in data.columns:
            data['FT_PCT'] = (data['FTM'] / data['FTA'] * 100).round(1)
        
    except Exception as e:
        print(f"\n❌ ERRO ao carregar dados: {str(e)}")
        print(f"\n💡 Verifique se o arquivo '{csv_file}' está correto.")
        print("   Execute: python download_bulls_data.py")
        return
    
    print("\n✅ Dados REAIS validados e prontos para análise!")
    
    # ETAPA 2: Regressão Linear - Pontos
    print("\n[2/7] Regressão Linear - Prevendo PONTOS (PTS)...")
    linear_pts = LinearRegressionAnalysis(data)
    linear_pts.prepare_data('PTS', ['FGM', 'FG3M', 'FTM'])
    linear_pts.train_model()
    metrics_pts = linear_pts.evaluate_model()
    print("\nGerando gráficos (Pontos)...")
    linear_pts.generate_all_plots()
    
    # ETAPA 3: Regressão Linear - Rebotes
    print("\n[3/7] Regressão Linear - Prevendo REBOTES (REB)...")
    linear_reb = LinearRegressionAnalysis(data)
    linear_reb.prepare_data('REB', ['FGA', 'FTA', 'STL'])
    linear_reb.train_model()
    metrics_reb = linear_reb.evaluate_model()
    
    # ETAPA 4: Regressão Linear - Assistências
    print("\n[4/7] Regressão Linear - Prevendo ASSISTÊNCIAS (AST)...")
    linear_ast = LinearRegressionAnalysis(data)
    linear_ast.prepare_data('AST', ['PTS', 'FGM', 'TOV'])
    linear_ast.train_model()
    metrics_ast = linear_ast.evaluate_model()
    
    # ETAPA 5: Regressão Logística
    print("\n[5/7] Regressão Logística - Prevendo VITÓRIA/DERROTA...")
    logistic_win = LogisticRegressionAnalysis(data)
    logistic_win.prepare_data('WL', ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'TOV'])
    logistic_win.train_model()
    metrics_win = logistic_win.evaluate_model()
    print("\nGerando gráficos (Regressão Logística)...")
    logistic_win.generate_all_plots()
    
    # ETAPA 6: Previsões de Probabilidade
    print("\n[6/7] Fazendo previsões de probabilidade de vitória...")
    
    # Calcular médias
    avg_stats = {
        'PTS': data['PTS'].mean(),
        'FG_PCT': data['FG_PCT'].mean(),
        'FG3_PCT': data['FG3_PCT'].mean(),
        'REB': data['REB'].mean(),
        'AST': data['AST'].mean(),
        'STL': data['STL'].mean(),
        'TOV': data['TOV'].mean()
    }
    
    print("\nCenário 1: Estatísticas MÉDIAS dos Bulls")
    X_avg = np.array([[avg_stats['PTS'], avg_stats['FG_PCT'], avg_stats['FG3_PCT'], 
                       avg_stats['REB'], avg_stats['AST'], avg_stats['STL'], avg_stats['TOV']]])
    logistic_win.predict_probability(X_avg, "Chicago Bulls", "Oponente Médio")
    
    print("\nCenário 2: Estatísticas ACIMA da média (bom desempenho)")
    X_good = np.array([[avg_stats['PTS']*1.1, avg_stats['FG_PCT']*1.05, avg_stats['FG3_PCT']*1.05,
                        avg_stats['REB']*1.1, avg_stats['AST']*1.1, avg_stats['STL']*1.1, avg_stats['TOV']*0.9]])
    logistic_win.predict_probability(X_good, "Chicago Bulls", "Oponente Fraco")
    
    print("\nCenário 3: Estatísticas ABAIXO da média (mau desempenho)")
    X_bad = np.array([[avg_stats['PTS']*0.9, avg_stats['FG_PCT']*0.95, avg_stats['FG3_PCT']*0.95,
                       avg_stats['REB']*0.9, avg_stats['AST']*0.9, avg_stats['STL']*0.9, avg_stats['TOV']*1.1]])
    logistic_win.predict_probability(X_bad, "Chicago Bulls", "Oponente Forte")
    
    # ETAPA 7: Relatório Final
    print("\n[7/7] Gerando relatório final...")
    print("\n" + "=" * 80)
    print("RELATÓRIO FINAL - CHICAGO BULLS")
    print("=" * 80)
    
    print(f"\nDados: {len(data)} jogos | Record: {sum(data['WL'] == 'W')}-{sum(data['WL'] == 'L')}")
    print(f"Média: {data['PTS'].mean():.1f} PPG | FG {data['FG_PCT'].mean():.1f}% | 3P {data['FG3_PCT'].mean():.1f}%")
    
    print("\n--- REGRESSÃO LINEAR ---")
    print(f"\n1. PONTOS: R²={metrics_pts['r2']:.4f}, RMSE={metrics_pts['rmse']:.2f}")
    print(f"   PTS = {metrics_pts['intercept']:.2f}", end="")
    for f, c in metrics_pts['coefficients'].items():
        print(f" {'+' if c>=0 else ''}{c:.2f}×{f}", end="")
    
    print(f"\n\n2. REBOTES: R²={metrics_reb['r2']:.4f}, RMSE={metrics_reb['rmse']:.2f}")
    print(f"   REB = {metrics_reb['intercept']:.2f}", end="")
    for f, c in metrics_reb['coefficients'].items():
        print(f" {'+' if c>=0 else ''}{c:.2f}×{f}", end="")
    
    print(f"\n\n3. ASSISTÊNCIAS: R²={metrics_ast['r2']:.4f}, RMSE={metrics_ast['rmse']:.2f}")
    print(f"   AST = {metrics_ast['intercept']:.2f}", end="")
    for f, c in metrics_ast['coefficients'].items():
        print(f" {'+' if c>=0 else ''}{c:.2f}×{f}", end="")
    
    print("\n\n--- REGRESSÃO LOGÍSTICA ---")
    print(f"\nVITÓRIA: Acurácia={metrics_win['accuracy']:.2%}, AUC-ROC={metrics_win['auc_roc']:.4f}")
    print(f"p(Vitória) = 1/[1 + e^-(z)], onde:")
    print(f"z = {metrics_win['intercept']:.4f}", end="")
    for f, c in metrics_win['coefficients'].items():
        print(f" {'+' if c>=0 else ''}{c:.4f}×{f}", end="")
    
    print("\n\nVariáveis mais importantes:")
    coeffs = sorted(metrics_win['coefficients'].items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (f, c) in enumerate(coeffs[:3], 1):
        print(f"  {i}. {f}: {'AUMENTA' if c>0 else 'DIMINUI'} chance de vitória (|coef|={abs(c):.4f})")
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print("\nGráficos salvos:")
    print("  - scatter_regression.png, prediction_vs_reality.png")
    print("  - confusion_matrix.png, trend_confidence.png")
    print("  - logistic_scatter_sigmoid.png, roc_curve.png")
    print("  - predicted_probabilities.png, feature_importance.png")
    print("  - logistic_confusion_matrix.png, logistic_trend_confidence.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Verificar se as bibliotecas necessárias estão instaladas
    print("Verificando dependências...")
    
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        print("Bibliotecas básicas instaladas!")
    except ImportError as e:
        print(f"ERRO: Biblioteca não encontrada: {e}")
        print("Instale as dependências com:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn")
        exit(1)
    
    try:
        import nba_api
        print("nba_api instalada!")
    except ImportError:
        print("AVISO: nba_api não encontrada (será usada geração de dados de exemplo)")
        print("Para baixar dados reais, instale:")
        print("   pip install nba_api")
    
    print("\n" + "=" * 70)
    
    # Perguntar se quer análise automática ou menu interativo
    print("\nModo de operação:")
    print("1. Análise AUTOMÁTICA completa dos Chicago Bulls")
    print("2. Menu INTERATIVO (escolher manualmente)")
    
    try:
        modo = input("\nEscolha (1 ou 2): ").strip()
        
        if modo == '1':
            executar_analise_completa_bulls()
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nOperação cancelada.")
    except Exception as e:
        print(f"\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()
