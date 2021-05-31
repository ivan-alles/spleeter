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


def separate(
    filename,
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
        'spleeter:2stems', MWF=False, stft_backend=STFTBackend.AUTO
    )
    separator.separate_to_file(
        filename,
        'output',
        audio_adapter=audio_adapter,
        synchronous=False,
    )
    separator.join()

def run(audio_file):
    separate(audio_file)

if __name__ == '__main__':
    run(sys.argv[1])
