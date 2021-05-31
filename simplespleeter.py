"""
 Simplified version without training and architectural sugar.
"""

import os
import json

import numpy as np
import librosa
import soundfile
from librosa.core import istft, stft
from scipy.signal.windows import hann

from functools import partial
from typing import Any, Dict, Iterable, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf
from tensorflow.compat.v1 import logging
from tensorflow.compat.v1.keras.initializers import he_uniform
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax,
)

placeholder = tf.compat.v1.placeholder

from spleeter.utils.tensor import pad_and_partition, pad_and_reshape



CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'pretrained_models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

def _get_conv_activation_layer(params: Dict) -> Any:
    """
    > To be documented.

    Parameters:
        params (Dict):

    Returns:
        Any:
            Required Activation function.
    """
    conv_activation: str = params.get("conv_activation")
    if conv_activation == "ReLU":
        return ReLU()
    elif conv_activation == "ELU":
        return ELU()
    return LeakyReLU(0.2)

def _get_deconv_activation_layer(params: Dict) -> Any:
    """
    > To be documented.

    Parameters:
        params (Dict):

    Returns:
        Any:
            Required Activation function.
    """
    deconv_activation: str = params.get("deconv_activation")
    if deconv_activation == "LeakyReLU":
        return LeakyReLU(0.2)
    elif deconv_activation == "ELU":
        return ELU()
    return ReLU()


def apply_unet(
    input_tensor: tf.Tensor,
    output_name: str = "output",
    params: Optional[Dict] = None,
    output_mask_logit: bool = False,
) -> Any:
    """
    Apply a convolutionnal U-net to model a single instrument (one U-net
    is used for each instrument).

    Parameters:
        input_tensor (tensorflow.Tensor):
        output_name (str):
        params (Optional[Dict]):
        output_mask_logit (bool):
    """
    logging.info(f"Apply unet for {output_name}")
    conv_n_filters = params.get("conv_n_filters", [16, 32, 64, 128, 256, 512])
    conv_activation_layer = _get_conv_activation_layer(params)
    deconv_activation_layer = _get_deconv_activation_layer(params)
    kernel_initializer = he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D, strides=(2, 2), padding="same", kernel_initializer=kernel_initializer
    )
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)
    batch6 = BatchNormalization(axis=-1)(conv6)
    _ = conv_activation_layer(batch6)
    #
    #
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((conv6))
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(0.5)(batch7)
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(0.5)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(0.5)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.
    if not output_mask_logit:
        up7 = Conv2D(
            2,
            (4, 4),
            dilation_rate=(2, 2),
            activation="sigmoid",
            padding="same",
            kernel_initializer=kernel_initializer,
        )((batch12))
        output = Multiply(name=output_name)([up7, input_tensor])
        return output
    return Conv2D(
        2,
        (4, 4),
        dilation_rate=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )((batch12))


class Separator:
    EPSILON = 1e-10

    def __init__(self, model_name):
        with open(os.path.join(CONFIG_DIR, model_name, 'base_config.json')) as f:
            self._params = json.load(f)

        self._sample_rate = self._params["sample_rate"]
        self._tf_graph = tf.Graph()
        self._prediction_generator = None
        self._features = None
        self._session = None

        self.stft_input_name = "{}_stft".format(self._params["mix_name"])

    def __del__(self) -> None:
        if self._session:
            self._session.close()

    @property
    def input_names(self):
        return ["audio_id", self.stft_input_name]

    def get_input_dict_placeholders(self):
        features = {
            self.stft_input_name: placeholder(
                tf.complex64,
                shape=(
                    None,
                    self._params["frame_length"] // 2 + 1,
                    self._params["n_channels"],
                ),
                name=self.stft_input_name,
            ),
            "audio_id": placeholder(tf.string, name="audio_id"),
        }
        return features

    def get_feed_dict(self, features, stft, audio_id):
        return


    def _stft(
        self, data: np.ndarray, inverse: bool = False, length: Optional[int] = None
    ) -> np.ndarray:
        """
        Single entrypoint for both stft and istft. This computes stft and
        istft with librosa on stereo data. The two channels are processed
        separately and are concatenated together in the result. The
        expected input formats are: (n_samples, 2) for stft and (T, F, 2)
        for istft.

        Parameters:
            data (numpy.array):
                Array with either the waveform or the complex spectrogram
                depending on the parameter inverse
            inverse (bool):
                (Optional) Should a stft or an istft be computed.
            length (Optional[int]):

        Returns:
            numpy.ndarray:
                Stereo data as numpy array for the transform. The channels
                are stored in the last dimension.
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self._params["frame_length"]
        H = self._params["frame_step"]
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": None} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = (
                np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
                if not inverse
                else data[:, :, c].T
            )
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            if inverse:
                s = s[N : N + length]
            s = np.expand_dims(s.T, 2 - inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2 - inverse)

    def _build_stft_feature(self):
        """Compute STFT of waveform and slice the STFT in segment
        with the right length to feed the network.
        """

        stft_name = 'mix_stft'
        spec_name = 'mix_spectrogram'

        if stft_name not in self._features:
            raise NotImplementedError('This code was never used.')

        if spec_name not in self._features:
            self._features[spec_name] = tf.abs(
                pad_and_partition(self._features[stft_name], self._params['T'])
            )[:, :, : self._params['F'], :]

    def _extend_mask(self, mask):
        """Extend mask, from reduced number of frequency bin to the number of
        frequency bin in the STFT.

        :param mask: restricted mask
        :returns: extended mask
        :raise ValueError: If invalid mask_extension parameter is set.
        """
        extension = self._params["mask_extension"]
        # Extend with average
        # (dispatch according to energy in the processed band)
        if extension == "average":
            extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
        # Extend with 0
        # (avoid extension artifacts but not conservative separation)
        elif extension == "zeros":
            mask_shape = tf.shape(mask)
            extension_row = tf.zeros((mask_shape[0], mask_shape[1], 1, mask_shape[-1]))
        else:
            raise ValueError(f"Invalid mask_extension parameter {extension}")
        n_extra_row = self._params['frame_length'] // 2 + 1 - self._params['F']
        extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
        return tf.concat([mask, extension], axis=2)

    def _build_masks(self):
        """
        Compute masks from the output spectrograms of the model.
        :return:
        """
        input_tensor = self._features['mix_spectrogram']

        output_dict = {}
        for instrument in self._params['instrument_list']:
            out_name = f"{instrument}_spectrogram"
            # outputs[out_name] = function(
            #     input_tensor, output_name=out_name, params=params or {}
            # )
            output_dict[out_name] = apply_unet(
                input_tensor, instrument, self._params["model"]["params"])

        separation_exponent = self._params["separation_exponent"]
        output_sum = (
            tf.reduce_sum(
                [e ** separation_exponent for e in output_dict.values()], axis=0
            )
            + self.EPSILON
        )
        out = {}

        for instrument in self._params['instrument_list']:
            output = output_dict[f"{instrument}_spectrogram"]
            # Compute mask with the model.
            instrument_mask = (
                output ** separation_exponent + (self.EPSILON / len(output_dict))
            ) / output_sum
            # Extend mask;
            instrument_mask = self._extend_mask(instrument_mask)
            # Stack back mask.
            old_shape = tf.shape(instrument_mask)
            new_shape = tf.concat(
                [[old_shape[0] * old_shape[1]], old_shape[2:]], axis=0
            )
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            # Remove padded part (for mask having the same size as STFT);

            stft_feature = self._features['mix_stft']
            instrument_mask = instrument_mask[: tf.shape(stft_feature)[0], ...]
            out[instrument] = instrument_mask
        self._masks = out

    def _get_session(self):
        if self._session is None:
            saver = tf.compat.v1.train.Saver()
            model_directory = os.path.join(MODEL_DIR, self._params["model_dir"])
            latest_checkpoint = tf.train.latest_checkpoint(model_directory)
            self._session = tf.compat.v1.Session()
            saver.restore(self._session, latest_checkpoint)
        return self._session

    def separate_waveform(self, waveform: np.ndarray):
        """
        Performs separation with librosa backend for STFT.

        Parameters:
            waveform (numpy.ndarray):
                Waveform to be separated (as a numpy array)
            audio_descriptor (AudioDescriptor):
        """
        with self._tf_graph.as_default():
            # out = {}
            self._features = self.get_input_dict_placeholders()

            self._build_stft_feature()

            self._build_masks()

            out = {}
            input_stft = self._features['mix_stft']
            for instrument, mask in self._masks.items():
                out[instrument] = tf.cast(mask, dtype=tf.complex64) * input_stft
            self._masked_stfts = out

            stft = self._stft(waveform)
            if stft.shape[-1] == 1:
                stft = np.concatenate([stft, stft], axis=-1)
            elif stft.shape[-1] > 2:
                stft = stft[:, :2]
            sess = self._get_session()
            feed_dict = {
                self._features["audio_id"]: 'my-audio',
                self._features[self.stft_input_name]: stft
            }
            outputs = sess.run(
                self._masked_stfts,
                feed_dict=feed_dict
            )
            for inst in self._params['instrument_list']:
                out[inst] = self._stft(
                    outputs[inst], inverse=True, length=waveform.shape[0]
                )
            return out

    def separate_file(self, audio_file):
        waveform, sr = librosa.load(audio_file, sr=self._sample_rate)
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform], 1)
        result = self.separate_waveform(waveform)
        fn, ext = os.path.splitext(os.path.basename(audio_file))
        out_dir = os.path.join(OUTPUT_DIR, fn)
        os.makedirs(out_dir, exist_ok=True)
        for instrument, output in result.items():
            soundfile.write(os.path.join(out_dir, instrument + '.wav'), output, self._sample_rate)

