from rl_games.common.algo_observer import AlgoObserver
import time
from collections import OrderedDict

import numpy as np
import torch
import random
import os


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2 ** attempt, 30))
                    attempt += 1

            return func(*args, **kwargs)
        return newfn
    return decorator




class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        wandb_unique_id = f'unique_id_{experiment_name}'
        print(f'Wandb using unique id {wandb_unique_id}')

        cfg = self.cfg

        print(cfg)

        # exit(0)

        # import ipdb; ipdb.set_trace()

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception, ))
        def init_wandb():
            wandb.init(
                project=cfg["wandb_project"],
                entity=cfg["wandb_entity"],
                group=cfg["wandb_group"],
                tags=cfg["wandb_tags"],
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=cfg["wandb_name"] + experiment_name,
                resume=True,
                dir=cfg["hydra"]["run"]["dir"],
                settings=wandb.Settings(start_method='fork'),
            )

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        wandb.config.update(config, allow_val_change=True)

