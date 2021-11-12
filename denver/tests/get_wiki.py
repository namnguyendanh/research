from pathlib import Path
from denver.trainers.language_model_trainer import LanguageModelTrainer

lm_trainer = LanguageModelTrainer()

lang = 'vi'
name = f"{lang}wiki"
data_path = Path('fastai')
path = data_path/name
path.mkdir(exist_ok=True, parents=True)
lm_trainer.get_wiki(path, lang, name)