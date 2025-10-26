"""
Script para baixar dados do Chicago Bulls - Temporada 2024-25
Usando nba_api com múltiplas tentativas
"""

import pandas as pd
from nba_api.stats.endpoints import teamgamelog, leaguegamefinder
from nba_api.stats.static import teams
import time

def download_chicago_bulls_2024_25():
    """
    Baixa dados completos do Chicago Bulls da temporada 2024-25
    Tenta múltiplos endpoints para garantir sucesso
    """
    print("=" * 70)
    print("DOWNLOAD DE DADOS - CHICAGO BULLS 2024-25")
    print("=" * 70)
    
    # Buscar ID do Chicago Bulls
    print("\n[1/4] Buscando informações do Chicago Bulls...")
    nba_teams = teams.get_teams()
    bulls = [team for team in nba_teams if team['full_name'] == 'Chicago Bulls'][0]
    team_id = bulls['id']
    print(f"✅ Time encontrado: {bulls['full_name']}")
    print(f"   ID: {team_id}")
    print(f"   Abreviação: {bulls['abbreviation']}")
    
    games_df = None
    
    # TENTATIVA 1: TeamGameLog
    print("\n[2/4] Tentativa 1: Baixando via TeamGameLog...")
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season='2024-25',
            season_type_all_star='Regular Season'
        )
        games_df = gamelog.get_data_frames()[0]
        
        if len(games_df) > 0:
            print(f"✅ {len(games_df)} jogos baixados via TeamGameLog!")
        else:
            print("⚠️ TeamGameLog retornou 0 jogos")
            games_df = None
            
    except Exception as e:
        print(f"❌ Erro no TeamGameLog: {str(e)}")
        games_df = None
    
    time.sleep(1)
    
    # TENTATIVA 2: LeagueGameFinder (se Tentativa 1 falhou)
    if games_df is None or len(games_df) == 0:
        print("\n[3/4] Tentativa 2: Baixando via LeagueGameFinder...")
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2024-25',
                season_type_nullable='Regular Season'
            )
            games_df = gamefinder.get_data_frames()[0]
            
            if len(games_df) > 0:
                print(f"✅ {len(games_df)} jogos baixados via LeagueGameFinder!")
            else:
                print("⚠️ LeagueGameFinder retornou 0 jogos")
                games_df = None
                
        except Exception as e:
            print(f"❌ Erro no LeagueGameFinder: {str(e)}")
            games_df = None
    
    # Processar e salvar dados se encontrados
    if games_df is not None and len(games_df) > 0:
        print(f"\n� Dados encontrados! Processando {len(games_df)} jogos...")
        
        # Garantir que temos as colunas necessárias
        required_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'PTS']
        if all(col in games_df.columns for col in required_cols):
            print(f"✅ Todas as colunas necessárias presentes")
            print(f"\n📋 Primeiros 5 jogos:")
            display_cols = [col for col in ['Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST'] 
                          if col in games_df.columns]
            print(games_df[display_cols].head())
            
            # Salvar em CSV
            filename = 'chicago_bulls_2024_25_games.csv'
            games_df.to_csv(filename, index=False)
            print(f"\n💾 Dados salvos em: {filename}")
            print(f"   Total de colunas: {len(games_df.columns)}")
            print(f"   Variáveis disponíveis: {', '.join(games_df.columns[:10])}...")
        else:
            print("⚠️ Algumas colunas necessárias estão faltando")
            print(f"   Colunas disponíveis: {', '.join(games_df.columns)}")
    else:
        print("\n⚠️ AVISO: Nenhum jogo encontrado para a temporada 2024-25")
        print("   Possíveis razões:")
        print("   1. A temporada ainda não começou oficialmente")
        print("   2. A API ainda não disponibilizou os dados")
        print("   3. Os dados estão em outro formato/endpoint")
        print("\n💡 Sugestão: O sistema usará dados simulados baseados em projeções")
    
    time.sleep(1)
    
    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO DO DOWNLOAD")
    print("=" * 70)
    
    if games_df is not None and len(games_df) > 0:
        wins = len(games_df[games_df['WL'] == 'W']) if 'WL' in games_df.columns else 0
        losses = len(games_df[games_df['WL'] == 'L']) if 'WL' in games_df.columns else 0
        
        print(f"✅ SUCESSO: {len(games_df)} jogos baixados")
        print(f"   Arquivo: chicago_bulls_2024_25_games.csv")
        if wins > 0 or losses > 0:
            print(f"   Record: {wins}W - {losses}L ({wins/(wins+losses)*100:.1f}%)")
        print(f"\n📊 Use no projeto:")
        print(f"   df = pd.read_csv('chicago_bulls_2024_25_games.csv')")
    else:
        print("⚠️ Nenhum dado disponível")
        print("   O projeto usará dados simulados automaticamente")
    
    print("=" * 70)
    
    return games_df


if __name__ == "__main__":
    download_chicago_bulls_2024_25()
