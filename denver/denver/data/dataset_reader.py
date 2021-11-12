# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import os
import logging
import pandas as pd

from overrides import overrides
from typing import Dict, List, Any
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MetadataField, Field

logger = logging.getLogger(__name__)

@DatasetReader.register("onenet")
class OneNetDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line
    and converts it into a ``Dataset`` suitable for sequence tagging. 
    Parameters
    ----------
    """
    def __init__(self,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_delimiter = token_delimiter

    def get_spans(self, words, tags):
        start = 0
        entity = None
        end = 0
        ping = False
        spans = []

        for i in range(len(tags)):
            if ping == True:
                if tags[i] == 'O':
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    ping = False

                elif ("B-" in tags[i]) and (i == len(tags) - 1):
                    # append the current span tags
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    start = i

                    # append the lastest span tags
                    entity = tags[i][2:]
                    end = i + 1
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))

                elif "B-" in tags[i]:
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    ping = True
                    start = i
                    entity = tags[i][2:]

                elif i == len(tags) - 1:
                    end = i+1
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))

            else:
                if "B-" in tags[i] and i == len(tags) - 1:
                    start = i
                    entity = tags[i][2:]
                    end = i + 1
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))

                elif "B-" in tags[i]:
                    start = i
                    entity = tags[i][2:]
                    ping = True
        return spans

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = os.path.abspath(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)

        nlu_samples = pd.read_csv(file_path, encoding='utf-8')

        for i in range(len(nlu_samples)):
            tokens = nlu_samples['text'][i].split()
            intent = nlu_samples['intent'][i]
            tags = nlu_samples['tag'][i].split()

            span_tags = self.get_spans(tokens, tags)

            if len(tokens) != len(tags):
                print("\n")
                logger.warning(f"[WARNING] Sample {i}: length of TOKENS different to length of TAGS. \n" 
                                f"TEXT ({len(nlu_samples['text'][i].split())}): {nlu_samples['text'][i]} \n"
                                f"TAG ({len(nlu_samples['tag'][i].split())}) : {nlu_samples['tag'][i]}")
                print("\n")
            else:

                tokens = [Token(token) for token in tokens]

                # nlu = intent + "#" + nlu_samples['tag'][i].strip()
                nlu = {intent: span_tags}

                yield self.text_to_instance(tokens, tags, span_tags, intent, nlu)

    def text_to_instance(
        self, 
        tokens: List[Token], 
        tags: List[str]=None, 
        span_tags: List[tuple]=None, 
        intent: str=None, 
        nlu: List[Any]=None
    ) -> Instance: # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)

        fields["tokens"] = sequence
        if tags:
            fields["tags"] = SequenceLabelField(tags, sequence)
        if intent:
            fields["intent"] = LabelField(intent, label_namespace="intent_labels")

        fields["metadata"] = MetadataField({
            'words': [x.text for x in tokens], 
            'intent': intent, 
            'tags': tags, 
            'span_tags': span_tags, 
            'nlu': nlu
        })

        return Instance(fields)