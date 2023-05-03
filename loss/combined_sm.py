from loss.implicit import get_implicit_score_matching
from loss.dsm import get_denosing_score_matching
from loss.yangsong import get_yang_song

def get_combined(cfg):

    ism = get_implicit_score_matching(cfg)
    dsm = get_denosing_score_matching(cfg)
    yang_song = get_yang_song(cfg)
    return lambda func, function_parameters, data, pertubred_data, time, _z, key: yang_song(func, function_parameters, data, pertubred_data, time, _z, key) + 0.001 * dsm(func, function_parameters, data, pertubred_data, time, _z, key)
    #return lambda func, function_parameters, data, pertubred_data, time, _z, key: ism(func, function_parameters, data, pertubred_data, time, _z, key) + dsm(func, function_parameters, data, pertubred_data, time, _z, key)