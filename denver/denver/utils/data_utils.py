# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import json
import logging

logger = logging.getLogger(__name__)

def save_json(data_fp, data, sort_keys=True, indent=4):
    with open(data_fp, 'w') as output_file:
        json.dump(data, output_file, sort_keys=sort_keys, indent=indent)