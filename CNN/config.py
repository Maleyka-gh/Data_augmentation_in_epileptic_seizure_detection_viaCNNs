import os
import config_ignored

# root for whole project
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


k = 2.4
list = []
# Data path
DATA_PATH = os.path.join(config_ignored.DATA_PATH, "cosinuss")
RAW_DF = os.path.join(config_ignored.DATA_PATH, "WORK/raw_Dataframes")
RAW_DF_FOLDER = os.path.join(config_ignored.DATA_PATH, "WORK/raw_Dataframes_sub_Dirs_new")
LEARN_DF = os.path.join(config_ignored.DATA_PATH, "WORK/dataframes_learn_new")
LEARN_NP_ALL = os.path.join(config_ignored.DATA_PATH, "learn_np_all")
LEARN_10_NoOverlap = os.path.join(config_ignored.DATA_PATH, "WORK/numpy_learn_10s_NoOverlap")
# LEARN_10_NoOverlapmain = os.path.join(config_ignored.DATA_PATH, "WORK/numpy_learn_10s_NoOverlap_main")
LEARN_10_Overlap50 = os.path.join(config_ignored.DATA_PATH, "numpy_learn_10s_Overlap50")
LEARN_10_NoOverlap_01 = os.path.join(config_ignored.DATA_PATH, "WORK/numpy_learn_10s_NoOverlap_k01_label10")
LEARN_30_NoOverlap = os.path.join(config_ignored.DATA_PATH, "numpy_learn_30s_NoOverlap")
LEARN_30_Overlap50 = os.path.join(config_ignored.DATA_PATH, "numpy_learn_30s_Overlap50")
LEARN_60_NoOverlap = os.path.join(config_ignored.DATA_PATH, "numpy_learn_60s_NoOverlap")
LEARN_60_Overlap50 = os.path.join(config_ignored.DATA_PATH, "numpy_learn_60s_Overlap50")
TEST_DF = os.path.join(config_ignored.DATA_PATH, "test_Dataframes")
OVERVIEW_DF = os.path.join(config_ignored.DATA_PATH, "overview_Dataframes")
DUPLICATES = os.path.join(config_ignored.DATA_PATH, "duplicates")
UCR = os.path.join(config_ignored.DATA_PATH, "UCR")
TRANSFER = os.path.join(config_ignored.DATA_PATH, "transfer_models")
AAA = os.path.join(config_ignored.DATA_PATH, "AAA")

# CNN MODELS
MODEL = os.path.join(config_ignored.DATA_PATH ,"ttsgan_results")