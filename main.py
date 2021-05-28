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
    deprecated_files: Optional[str] = AudioInputOption,
    files: List[Path] = AudioInputArgument,
    adapter: str = AudioAdapterOption,
    bitrate: str = AudioBitrateOption,
    codec: Codec = AudioCodecOption,
    duration: float = AudioDurationOption,
    offset: float = AudioOffsetOption,
    output_path: Path = AudioOutputOption,
    stft_backend: STFTBackend = AudioSTFTBackendOption,
    filename_format: str = FilenameFormatOption,
    params_filename: str = ModelParametersOption,
    mwf: bool = MWFOption,
    verbose: bool = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from .audio.adapter import AudioAdapter
    from .separator import Separator

    configure_logger(verbose)
    if deprecated_files is not None:
        logger.error(
            "⚠️ -i option is not supported anymore, audio files must be supplied "
            "using input argument instead (see spleeter separate --help)"
        )
        raise Exit(20)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(
        params_filename, MWF=mwf, stft_backend=stft_backend
    )
    for filename in files:
        separator.separate_to_file(
            str(filename),
            str(output_path),
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False,
        )
    separator.join()

def run(audio_file):
    print(audio_file)

if __name__ == '__main__':
    run(sys.argv[1])