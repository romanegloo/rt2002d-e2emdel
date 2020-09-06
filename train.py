"""
train.py

Main script for training and saving E2E MD-EL models for PubTator annotation.
Modify the configuration file () for experiments.
"""

import logging
import configparser
import code

from E2EMDEL.data import KaggleNERDataset
from E2EMDEL.model import MDEL

logger = logging.getLogger(__name__)
conf = configparser.ConfigParser()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )
    logger.info('Reading the configuration file')
    conf.read('conf.ini')

    ds = KaggleNERDataset()
    model = MDEL()

    code.interact(local=locals())