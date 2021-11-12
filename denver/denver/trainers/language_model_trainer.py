# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import re
import logging

from pathlib import Path
from pandas import DataFrame
from typing import Union, List, Text
from fastai.basics import download_url, bunzip, shutil, working_directory
from fastai.text import TextList, language_model_learner, AWD_LSTM, Path, TextLMDataBunch

from denver import DENVER_DIR
from denver.utils.print_utils import print_line
from denver.utils.utils import download_url as dl

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class LanguageModelTrainer:    
    """A class Language model trainer. """

    def __init__(self, pretrain: Union[str, Text, List[str], List[Path]]='babe'):
        """Initialize a class Language model trainer

        :param pretrain: The name of the pretrained language model ('wiki', 'babe') or None value
                           or the path to your pretrained language model.
        """
    
        if pretrain == 'babe':
            urls = {
                'vi_wt_babe.pth': 'http://minio.dev.ftech.ai/fastaibase-v0.0.3-677cd84c/vi_wt_babe.pth', 
                'vi_wt_vocab_babe.pkl': 'http://minio.dev.ftech.ai/fastaibase-v0.0.3-677cd84c/vi_wt_vocab_babe.pkl'
            }

            self.lm_fns_pretrained = self.downdload_lm(urls)
            logger.debug(f"Load the pretrained language-model from: {self.lm_fns_pretrained}")

        elif pretrain == 'wiki':
            urls = {
                'vi_wt_wiki.pth': 'http://minio.dev.ftech.ai/fastaibase-v0.0.4-f17ecdb1/vi_wt_wiki.pth', 
                'vi_wt_vocab_wiki.pkl': 'http://minio.dev.ftech.ai/fastaibase-v0.0.4-f17ecdb1/vi_wt_vocab_wiki.pkl'
            }

            self.lm_fns_pretrained = self.downdload_lm(urls)
            logger.debug(f"Load the pretrained language model from {self.lm_fns_pretrained}")

        elif pretrain == None:
            self.lm_fns_pretrained == None
            logger.warning(f"The pretrained language model is a `None` value.")

        elif isinstance(pretrain, List):
            lm_fns_pretrained = [os.path.abspath(p) for p in pretrain]

            if not os.path.exists(lm_fns_pretrained[0]+'.pth') or os.path.exists(lm_fns_pretrained[1]+'pkl'):
                logger.error(f"The `{lm_fns_pretrained}` is not found or not supplied!")
                logger.info(f"Setup `lm_pretrained_path` into `None` value !")
                
                self.lm_fns_pretrained = None
            else:
                self.lm_fns_pretrained = lm_fns_pretrained

        else:
            self.lm_fns_pretrained = None
            logger.warning(f"`lm_fns_pretrained` must be in ['babe', 'wiki', None] or List of paths to pretrained language model. "
                        f"Setup lm_fns_pretrained=None")

    def downdload_lm(self, urls):
        """Download the pretrained language models. """
        lm_fns_pretrained = []
        dest = f'{DENVER_DIR}'

        for name, url in urls.items():
            dl(url, dest, name)
            lm_fns_pretrained.append(os.path.abspath(dest + name.split('.')[0]))
        return lm_fns_pretrained

    def get_wiki(self, path, lang, name):
        '''Wikipedia data download function

        :param path: Path to save the dataset.
        :param lang: Language wikipedia. Example: vi, en,..
        :param name: The directory name that stores the dataset.
        '''
        path = Path(path)
        if (path/name).exists():
            logger.info(f"  {path/name} already exists; not downloading")
            return

        xml_fn = f"{lang}wiki-latest-pages-articles.xml"
        zip_fn = f"{xml_fn}.bz2"

        if not (path/xml_fn).exists():
            logger.info(f"Downloading...")
            download_url(f'https://dumps.wikimedia.org/{name}/latest/{zip_fn}', path/zip_fn)
            logger.info(f"Unzipping...")
            bunzip(path/zip_fn)

        with working_directory(path):
            if not (path/'wikiextractor').exists(): os.system('git clone https://github.com/attardi/wikiextractor.git')
            logger.info(f"Extracting...")
            os.system("python wikiextractor/WikiExtractor.py --processes 4 --no_templates " +
                f"--min_text_length 1800 --filter_disambig_pages --log_file log -b 100G -q {xml_fn}")
        shutil.move(str(path/'text/AA/wiki_00'), str(path/name))
        shutil.rmtree(path/'text')

    def split_wiki(self, path, lang, name):
        '''Split data into sub-files

        :param path: Path to save the dataset.
        :param lang: Language wikipedia. Example: vi, en,..
        :param name: The directory name that stores the dataset.
        '''
        path = Path(path)
        dest = path/'docs'
        if dest.exists():
            logger.info(f"{dest} already exists; not splitting")
            return dest

        dest.mkdir(exist_ok=True, parents=True)
        title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
        lines = (path/name).open()
        f=None

        for i,l in enumerate(lines):
            if i%100000 == 0: print(i)
            if l.startswith('<doc id="'):
                title = title_re.findall(l)[0].replace('/','_')
                if len(title)>150: continue
                if f: f.close()
                f = (dest/f'{title}.txt').open('w')
            else: f.write(l)
        f.close()
        return dest

    def fine_tuning_from_wiki(
        self, 
        lm_fns: Union[List[str], List[Path]]=None, 
        lang: str='vi',
        seed: int=42, 
        num_workers: int=1,
        num_epochs: int=10,
        learning_rate: float=1e-3,
        moms: Union[tuple, List[float]]=(0.8, 0.7),
        batch_size: int=128,
        drop_mult: float=0.5,
        pct: float=0.1,
    ):
        """Fine-tunning Language model with Wiki dataset based on AWD_LSTM architecture.

        :param lm_fns: The path to save language model
        :param lang: The name of the language, default='vi
        :param seed: The number of seed
        :param num_wokers: The number of workers
        :param num_epochs: The number of epochs
        :param learning_rate: The learning rate
        :param moms: The momentums
        :param batch_size: The batch size
        :param drop_mult: The dropout multiple
        :param pct: The ratio to split train and valid during when training 
        """
        name = f'{lang}wiki'
        data_path = Path('fastai')
        path = data_path/name
        path.mkdir(exist_ok=True, parents=True)

        self.get_wiki(path, lang, name)
        dest = self.split_wiki(path, lang, name)

        data_lm = TextList.from_folder(dest).split_by_rand_pct(pct, seed=seed).label_for_lm() \
                          .databunch(bs=batch_size, num_workers=num_workers)
        data_lm.save(f'{lang}_databunch')

        learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=drop_mult, pretrained=self.lm_fns_pretrained)
        
        _lr = learning_rate * batch_size / 48     #Scale learning rate by batchsize

        learn.unfreeze()
        logger.info(f"Training language model with Wiki corpus...")
        
        learn.fit_one_cycle(num_epochs, _lr, moms=moms)

        if lm_fns is not None:
            lm_fns = [os.path.abspath(p) for p in lm_fns]
        elif self.lm_fns_pretrained is not None:
            lm_fns = self.lm_fns_pretrained
            logger.info(f"Save the fine-tuned language model into {lm_fns}")
        else:
            if not os.path.exists('./models/.finetune/'):
                os.makedirs('./models/.finetune/')

            lm_fns = ["./models/.finetune/vi_wt", "./models/.finetune/vi_wt_vocab"]
            lm_fns = [os.path.abspath(p) for p in lm_fns]
            logger.info(f"Save the fine-tuned language model into {lm_fns}")
        
        learn.to_fp32().save(lm_fns[0], with_opt=False)
        learn.data.vocab.save(lm_fns[1] + '.pkl')


    def fine_tuning_from_folder(
        self, 
        data_folder: Union[str, Path]=None, 
        lm_fns: Union[List[str], List[Path]]=None, 
        seed: int=42,
        num_workers: int=1,
        num_epochs: int=10,
        learning_rate: float=1e-2,
        exp_lr: float=5.75e-2,
        moms: Union[tuple, List[float]]=(0.8, 0.7),
        batch_size: int=48,
        drop_mult: float=0.5,
        pct: float=0.1,
    ):
        '''Fine-tuning language model with folder dataset for fine-tuning LM

        :param data_folder: the path to folder data
        :param lm_fns: The path to save language model
        :param seed: The number of seed
        :param num_wokers: The number of workers
        :param num_epochs: The number of epochs
        :param learning_rate: The learning rate
        :param exp_lr: The experiment learning rate for the best results
        :param moms: The momentums
        :param batch_size: The batch size
        :param drop_mult: The dropout multiple
        :param pct: The ratio to split train and valid during when training 
        '''
        data_lm = (TextList.from_folder(data_folder)
                .split_by_rand_pct(pct, seed=seed)
                .label_for_lm()
                .databunch(bs=batch_size, num_workers=num_workers))
        learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=self.lm_fns_pretrained, drop_mult=drop_mult)

        # 5.75e-02 is the experimental value for best results

        logger.info(f"Fine-tuning language model with data corpus: {data_folder}")

        learn_lm.fit_one_cycle(int(0.2*num_epochs), exp_lr, moms=moms)    

        learn_lm.unfreeze()
        learn_lm.fit_one_cycle(int(0.8*num_epochs), learning_rate, moms=moms)
        
        if lm_fns is not None:
            lm_fns = [os.path.abspath(p) for p in lm_fns]
        elif self.lm_fns_pretrained is not None:
            lm_fns = self.lm_fns_pretrained
            logger.info(f"Save the fine-tuned language model into {lm_fns}")
        else:
            if not os.path.exists('./models/.finetune/'):
                os.makedirs('./models/.finetune/')

            lm_fns = ["./models/.finetune/vi_wt", "./models/.finetune/vi_wt_vocab"]
            lm_fns = [os.path.abspath(p) for p in lm_fns]
            logger.info(f"Save the fine-tuned language model into {lm_fns}")

        learn_lm.to_fp32().save(lm_fns[0], with_opt=False)
        learn_lm.data.vocab.save(str(lm_fns[1]) + '.pkl')

    
    def fine_tuning_from_df(
        self, 
        data_df: DataFrame, 
        seed: int=42,
        num_workers: int=1,
        num_epochs: int=10,
        learning_rate: float=1e-3,
        moms: Union[tuple, List[float]]=[0.8, 0.7],
        batch_size: int=128,
        drop_mult: float=0.5,
    ):
        '''Fine-tuning language model with dataframe

        :param data_df: A train Dataframe
        :param seed: The number of seed
        :param num_wokers: The number of workers
        :param num_epochs: The number of epochs
        :param learning_rate: The learning rate
        :param moms: The momentums
        :param batch_size: The batch size
        :param drop_mult: The dropout multiple
        '''
        # df = df.iloc[np.random.permutation(len(df))]
        # cut = int(pct * len(df)) + 1
        # train_df, valid_df = df[cut:], df[:cut]

        data_lm = TextLMDataBunch.from_df(train_df=data_df, valid_df=data_df, path='', 
                                          text_cols='text', label_cols='label')

        learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=self.lm_fns_pretrained, drop_mult=drop_mult)

        print_line(text='FINE-TUNING')

        _lr = learning_rate * batch_size / 48

        learn_lm.fit_one_cycle(int(0.2*num_epochs), _lr * 10, moms=moms)

        learn_lm.unfreeze()
        learn_lm.fit_one_cycle(int(0.8*num_epochs), _lr, moms=moms)

        if not os.path.exists(f'{DENVER_DIR}'):
            os.makedirs(f'{DENVER_DIR}')
        
        lm_fns = [f"{DENVER_DIR}/vifine_tuned_ic", f"{DENVER_DIR}/vifine_tuned_enc_ic"]
        lm_fns = [os.path.abspath(p) for p in lm_fns]

        learn_lm.save(lm_fns[0])
        learn_lm.save_encoder(lm_fns[1])

        # # Save model
        # logger.info(f"\tPath to the saved encode language model: '{lm_fns[1]}'")
