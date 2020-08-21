# *_*coding:utf-8 *_*
import os

# path
PATH_TO_MUSE_2020 = '/data1/sunlicai/Affective Computing/Dataset/MuSe/2020' # change this path to yours

PATH_TO_ALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'c1_muse_wild/feature_segments/label_aligned')
PATH_TO_UNALIGNED_FEATURES = os.path.join(PATH_TO_MUSE_2020, 'c1_muse_wild/feature_segments/unaligned')
PATH_TO_LABELS = os.path.join(PATH_TO_MUSE_2020, 'c1_muse_wild/label_segments')
PATH_TO_TRANSCRIPTIONS = os.path.join(PATH_TO_MUSE_2020, 'c1_muse_wild/transcription_segments')

PATH_TO_METEDATA = os.path.join(PATH_TO_MUSE_2020, 'raw/metadata/data/processed_tasks/metadata')
PARTITION_FILE = os.path.join(PATH_TO_METEDATA, 'partition.csv')
META_FILE = os.path.join(PATH_TO_METEDATA, 'video_metadata.csv')

PATH_TO_RAW_AUDIO = os.path.join(PATH_TO_MUSE_2020, 'raw/data/raw/audio_norm')
PATH_TO_RAW_VIDEO = os.path.join(PATH_TO_MUSE_2020, 'raw/data/raw/video')

DATA_FOLDER = './data'
MODEL_FOLDER = './model'
LOG_FOLDER = './log'
PREDICTION_FOLDER = './prediction'
FUSION_FOLDER = './fusion'

# numerical
EPSILON = 1e-6