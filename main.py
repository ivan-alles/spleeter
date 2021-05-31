import sys

import json
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from pathlib import Path
from typing import Container, Dict, List, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from spleeter import SpleeterError
from spleeter.options import *
from spleeter.utils.logging import configure_logger, logger

import simplespleeter


def separate(
    filename,
    model,
    adapter: str = AudioAdapterOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from spleeter.audio.adapter import AudioAdapter
    from spleeter.separator import Separator

    configure_logger(verbose)

    audio_adapter: AudioAdapter = AudioAdapter.get(adapter.default)
    separator: Separator = Separator(
        'spleeter:' + model, MWF=False, stft_backend=STFTBackend.AUTO,
        multiprocess=False
    )
    separator.separate_to_file(
        filename,
        'output',
        audio_adapter=audio_adapter,
        synchronous=False,
    )
    separator.join()

def run(audio_file, model):
    if True:
        separate(audio_file, model)
    else:
        spleeter = simplespleeter.Separator(model)
        spleeter.separate_file(audio_file)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('Usage: main audio model')
        print('model: 2stems, 3stems, 4stems')
    else:
        run(sys.argv[1], sys.argv[2])
