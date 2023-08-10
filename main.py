
# Gato Mestre models
from gmodels.binary_classification import BinaryClassifier
from gmodels.cross_validation import CrossValidate
from gmodels.regression_valorizacao import FormulaRegression
from gmodels.regression import LinearRegression
from gmodels.xG import xG
from gmodels.nn import NeuralNetwork
# base Cartola
from base import Base
# database instance
from atualizar_redis import AtualizarRedis



xg = xG(2023)
#xg.gerar_metricas_xG()
clubes_xg, clubes_xg_mando, clubes_xg_diff, clubes_xG_probs, clubes_xg_defense, \
clubes_goals_mando, clubes_gols_diff, clubes_xg_potencial, \
atletas_xg, atletas_xg_detalhado, atletas_xg_per_match, \
atletas_xg_eficiencia, atletas_xg_potencial = xg.gerar_metricas_xG(torneio='Brasileiro',edicao=2023)