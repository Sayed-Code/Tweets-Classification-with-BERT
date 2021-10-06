from bert.utils import get_project_root

ROOT = str(get_project_root())
PATH_TRAINING_2020 = ROOT + '/data/2020/2020_full_German/exp1/train.xlsx'
PATH_VALIDATION_2020 = ROOT + '/data/2020/2020_full_German/exp1/validation.xlsx'

PATH_TRAINING_2019 = ROOT + '/data/2019/2019_English/train.xlsx'
PATH_VALIDATION_2019 = ROOT + '/data/2019/2019_English/validation.xlsx'

PATH_TRAINING_2019_2020 = ROOT + '/data/2019_2020/english/train.xlsx'
PATH_VALIDATION_2019_2020 = ROOT + '/data/2019_2020/english/validation.xlsx'

PATH_TESTING_2019_2020 = ROOT + '/data/2019_2020/english/test.xlsx'
PATH_TESTING_2020 = ROOT + '/data/2020/2020_full_German/exp1/test.xlsx'
PATH_TESTING_2019 = ROOT + '/data/2019/2019_English/test.xlsx'

PATH_SAVED_MODELS = ROOT + '/saved_models'