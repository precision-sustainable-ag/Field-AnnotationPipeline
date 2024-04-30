#!/usr/bin/env python3
import logging

from omegaconf import DictConfig

log = logging.getLogger(__name__)

class SomeClass:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.print_cfg()

    def print_cfg(self):
        print(self.cfg)


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    SomeClass(cfg)
    log.info(f"{cfg.general.task} completed.")
