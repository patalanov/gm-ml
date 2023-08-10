import redis
from redis_conector import RedisConnector
import json
from calls import get_equipes_sde, get_atletas_sde


class AtualizarRedis:

    def __init__(self, edicao_cartola):
        self.redis = RedisConnector()  # Acessa dinamicamente o redis de acordo com o ambiente
        # self.r = redis.Redis(host='localhost', port=6379, db=0)  # Local

        # Busca os dados de equipes e atletas mais recentes no SDE
        self.equipes_sde = get_equipes_sde(edicao_cartola)[2]
        self.atletas_sde = get_atletas_sde(edicao_cartola, self.equipes_sde.keys())

    def atualizar(self, tipo, nome, rodada, dados):
        """
        Armazena diretamente no Redis do Gato Mestre
        Args:
            tipo (str): "indices", "medias", "probabilidades", "confrontos", "distribuicoes", "atletas", etc
            nome (str): "eficiencia-ofensiva", "aproveitamento-ofensivo", etc
            rodada (int): 1, 2, 3, etc
            dados (object): Objeto formatado para o Redis
        """
        self.redis.set(f'{tipo}/{nome}/rodadas/{rodada}', dados)

    def atualizar_times(self, rodada, dados):
        self.redis.set(f'times/{rodada}', dados)

    def formatar_equipe(self, equipe_id):
        equipe = self.equipes_sde.get(equipe_id)

        if not equipe:
            return None

        return {
            'equipe_id': equipe['equipe_id'],
            'nome_popular': equipe.get('nome_popular', ''),
            'sigla': equipe.get('sigla', ''),
            'escudo': equipe['escudos']['svg'] or equipe['escudos']['60x60'] if equipe.get('escudos') else None,
            'cores': {
                'primaria': equipe['cores'].get('primaria', '') if equipe.get('cores') else '',
                'secundaria': equipe['cores'].get('secundaria', '') if equipe.get('cores') else '',
            }
        }

    def formatar_atleta(self, atleta_id):
        atleta = self.atletas_sde.get(str(atleta_id))

        if not atleta:
            return None

        return {
            'atleta_id': int(atleta_id),
            'nome_popular': atleta['nome_popular'],
            'foto': atleta['fotos']['140x140'] if atleta.get('fotos') else None,
            'posicao': {
                'sigla': atleta['posicao'].get('sigla', ''),
                'nome': atleta['posicao'].get('descricao', ''),
            }
        }


    def atualizar_base(self, dataframe, rodada):
        formatado_redis = []
        dataframe = dataframe[dataframe['rodada_id']==rodada]
        for index, _ in dataframe.get('rodada_id').items():
            formatado_redis.append({
                'nome': str(dataframe['nome'][index]),
                'slug': str(dataframe['slug'][index]),
                'apelido': str(dataframe['apelido'][index]),
                'atleta_id': int(dataframe['atleta_id'][index]),
                'rodada_id': int(dataframe['rodada_id'][index]),
                'clube_id': int(dataframe['clube_id'][index]),
                'posicao_id': int(dataframe['posicao_id'][index]),
                'status_id': int(dataframe['status_id'][index]),
                'pontos_num': float(dataframe['pontos_num'][index]),
                'preco_num': float(dataframe['preco_num'][index]),
                'variacao_num': float(dataframe['variacao_num'][index]),
                'media_num': float(dataframe['media_num'][index]),
                'jogos_num': int(dataframe['jogos_num'][index]),
                'clube': str(dataframe['clube'][index]),
                'posicao': str(dataframe['posicao'][index]),
                'status_pre': str(dataframe['status_pre'][index]),
                'minutos_jogados': str(dataframe['minutos_jogados'][index]),
                'duracao_partida': str(dataframe['duracao_partida'][index]),
                'duracao_1t': str(dataframe['duracao_1t'][index]),
                'duracao_2t': str(dataframe['duracao_2t'][index]),
                'pontos_por_minuto': str(dataframe['pontos_por_minuto'][index]),
                'FF': float(dataframe['FF'][index]),
                'FS': float(dataframe['FS'][index]),
                'G': float(dataframe['G'][index]),
                'A': float(dataframe['A'][index]),
                'PI': float(dataframe['PI'][index]),
                'SG': float(dataframe['SG'][index]),
                'DE': float(dataframe['DE'][index]),
                'DS': float(dataframe['DS'][index]),
                'FC': float(dataframe['FC'][index]),
                'GC': float(dataframe['GC'][index]),
                'GS': float(dataframe['GS'][index]),
                'FD': float(dataframe['FD'][index]),
                'CA': float(dataframe['CA'][index]),
                'FT': float(dataframe['FT'][index]),
                'I': float(dataframe['I'][index]),
                'PP': float(dataframe['PP'][index]),
                'CV': float(dataframe['CV'][index]),
                'PC': float(dataframe['PC'][index]),
                'PS': float(dataframe['PS'][index]),
                'preco_open': float(dataframe['preco_open'][index]),
                'ano': str(dataframe['ano'][index]),
                'home_dummy': int(dataframe['home_dummy'][index]),
                'opponent': int(dataframe['opponent'][index]),
                'team_goals': float(dataframe['team_goals'][index]),
                'opp_goals': float(dataframe['opp_goals'][index]),
                'match_id': int(dataframe['match_id'][index]),
                'adversario_nome': str(dataframe['adversario_nome'][index]),
                'DP': float(dataframe['DP'][index]),
                'entrou_em_campo': bool(dataframe['entrou_em_campo'][index]),
                'exibir_atleta': bool(dataframe['exibir_atleta'][index]),
            })

        self.atualizar('base', 'base_cartola', rodada, formatado_redis)


    def atualizar_calculadora(self, dataframe, rodada):
        formatado_redis = {}

        for index, atleta_id in dataframe.get('atleta_id').items():
            # TODO realizar requisição ao SDE e incluir esse atleta na lista geral
            # Verifica se o atleta foi encontrado
            if not self.atletas_sde.get(str(atleta_id)):
                print(f'\nERRO\t>>>>\tO Atleta {atleta_id} da Equipe {dataframe["clube_id"][index]} não foi '
                      f'encontrado nos vínculos recentes do SDE e não será salvo no Redis!')
                continue

            atleta = self.formatar_atleta(int(atleta_id))
            equipe = self.formatar_equipe(int(dataframe['clube_id'][index]))
            adversario = self.formatar_equipe(int(dataframe['adversario_id'][index]))

            formatado_redis[str(atleta_id)] = {
                'atleta': atleta,
                'equipe': equipe,
                'adversario': adversario,
                'posicao_id': int(dataframe['posicao_id'][index]),
                'partidas_jogadas': int(dataframe['jogos_num'][index]),
                'media_mandante': float(dataframe['media_casa'][index]),
                'media_visitante': float(dataframe['media_fora'][index]),
                'media_geral': float(dataframe['media_geral'][index]),
                'media_movel_mandante': float(dataframe['media_movel_casa'][index]),
                'media_movel_visitante': float(dataframe['media_movel_fora'][index]),
                'media_movel_geral': float(dataframe['media_movel_geral'][index]),
                'indice_eficiencia_ofensiva': float(dataframe['ieo_gm_mean'][index]),
                'indice_defensivo': float(dataframe['ied_gm_mean'][index]),
                'indice_goleiro': float(dataframe['ieg_gm_mean'][index]),
                'indice_pontos_cartoleta': float(dataframe['ippc_gm_mean'][index]),
                'indice_representacao_ofensiva': float(dataframe['iro_gm'][index]),
                'indice_representacao_pts': float(dataframe['irp_gm'][index]),
                'expectativa_pontos_mandante': float(dataframe['XP_mandante'][index]),
                'expectativa_pontos_visitante': float(dataframe['XP_visitante'][index]),
                'expectativa_gols_mandante': float(dataframe['XP_xG_mandante'][index]),
                'expectativa_gols_visitante': float(dataframe['XP_xG_visitante'][index]),
                'diff_expectativa_e_gols_feitos_clube': float(dataframe['xG_G_composite'][index]),
                'probabilidade_sg': float(dataframe['prob_SG'][index]),
                'probabilidade_pts_acima_media': float(dataframe['prob_pts_acima_media'][index]),
                'probabilidade_vitoria_mandante_xG':float(dataframe['prob_vitoria_mandante_xG'][index]),
                'probabilidade_vitoria_visitante_xG':float(dataframe['prob_vitoria_visitante_xG'][index]),
                'probabilidade_empate_xG':float(dataframe['prob_empate_xG'][index]),
                'probabilidade_sg_mandante_xG':float(dataframe['prob_sg_mandante_xG'][index]),
                'probabilidade_sg_visitante_xG':float(dataframe['prob_sg_visitante_xG'][index]),
                'probabilidade_sg_empate_xG':float(dataframe['prob_sg_empate_xG'][index]),
                'probabilidade_vitoria_mandante_combinada':float(dataframe['prob_mandante_media'][index]),
                'probabilidade_vitoria_visitante_combinada':float(dataframe['prob_visitante_media'][index]),
                'probabilidade_empate_combinada':float(dataframe['prob_empate_media'][index]),
                'probabilidade_sg_mandante_combinada':float(dataframe['prob_sg_mandante_media'][index]),
                'probabilidade_sg_mandante_combinada':float(dataframe['prob_sg_visitante_media'][index]),
                'expectativa_gols_atleta': float(dataframe['xG_atleta'][index]),
                'expectativa_gols_eficiencia_atleta': float(dataframe['xG_eficiencia_atleta'][index]),
                'minimo_pontos_para_valorizar': float(dataframe['XP_para_valorizar'][index]),
                'objetivo': str(dataframe['strategy'][index]),
                'formacao': str(dataframe['formation'][index]),
                'cartoletas': float(dataframe['budget'][index]),
                'preco_time': float(dataframe['preco_time'][index]),
                'capitao': int(dataframe['C'][index]),
            }

        self.atualizar_times(rodada, formatado_redis)

