import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import Batch, PiperDataset, UtteranceCollate
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss, subband_stft_loss
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import (
    SynthesizerTrn, 
    MultiPeriodDiscriminator,
    DurationDiscriminatorV1,
    DurationDiscriminatorV2,
    AVAILABLE_FLOW_TYPES,
    AVAILABLE_DURATION_DISCRIMINATOR_TYPES
)
from .pqmf import PQMF

_LOGGER = logging.getLogger("vits.lightning")


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        # audio
        resblock="1",
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=(
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        upsample_rates=(8, 8),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16),
        # Vits2
        use_mel_posterior_encoder: bool = True,
        use_duration_discriminator: bool = True,
        duration_discriminator_type: str = "dur_disc_2",
        use_transformer_flows: bool = True,
        transformer_flow_type: str = "pre_conv2",
        use_spk_conditioned_encoder: bool = False,
        use_noise_scaled_mas: bool = True,
        mas_noise_scale_initial: float = 0.01,
        noise_scale_delta = 2e-6,
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        ms_istft_vits: bool = False,
        mb_istft_vits: bool = False,
        istft_vits: bool = True,
        subbands: bool = False,
        gen_istft_n_fft: int = 16,
        gen_istft_hop_size: int = 4,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = False,
        segment_size: int = 8192,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        fft_sizes: Tuple[float, float, float] = (384, 683, 171),
        hop_sizes: Tuple[float, float, float] = (30, 60, 10),
        win_lengths: Tuple[float, float, float] = (150, 300, 60),
        window: str = "hann_window",
        grad_clip: Optional[float] = None,
        num_workers: int = 1,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512
        # vits2:
        if self.hparams.use_mel_posterior_encoder == True:  # P.incoder for vits2
            print("Using mel posterior encoder for VITS2")
            self.posterior_channels = 80  # vits2
        else:
            print("Using lin posterior encoder for VITS1")
            self.posterior_channels = self.hparams.filter_length // 2 + 1

        # More VITS2 features:
        if self.hparams.use_transformer_flows:
            self.transformer_flow_type = self.hparams.transformer_flow_type
            print(f"Using transformer flows {self.transformer_flow_type} for VITS2")
            assert self.transformer_flow_type in AVAILABLE_FLOW_TYPES, f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        else:
            print("Using normal flows for VITS1")

        if self.hparams.use_spk_conditioned_encoder:
            if self.hparams.num_speakers == 0:
                print("Warning: use_spk_conditioned_encoder is True but num_speakers is 0")
            print("Setting use_spk_conditioned_encoder to False as model is a single speaker model")
        else:
            print("Using normal encoder for VITS1 (cuz it's single speaker after all)")

        if self.hparams.use_noise_scaled_mas:
            print("Using noise scaled MAS for VITS2")
            self.mas_noise_scale_initial = 0.01
            self.noise_scale_delta = 2e-6
        else:
            print("Using normal MAS for VITS1")
            self.mas_noise_scale_initial = 0.0
            self.noise_scale_delta = 0.0


        # Set up models
        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.posterior_channels,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            gen_istft_n_fft=self.hparams.gen_istft_n_fft,
            gen_istft_hop_size=self.hparams.gen_istft_hop_size,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
            subbands=self.hparams.subbands,
            ms_istft_vits=self.hparams.ms_istft_vits,
            mb_istft_vits=self.hparams.mb_istft_vits,
            istft_vits=self.hparams.istft_vits,
            use_spk_conditioned_encoder =self.hparams.use_spk_conditioned_encoder,
            use_transformer_flows=self.hparams.use_transformer_flows,
            transformer_flow_type=self.hparams.transformer_flow_type,
            use_noise_scaled_mas=self.hparams.use_noise_scaled_mas,
            mas_noise_scale_initial=self.mas_noise_scale_initial,
            noise_scale_delta=self.noise_scale_delta,
        )
        self.model_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        )

        if self.hparams.use_duration_discriminator:
            # print("Using duration discriminator for VITS2")
            #- for duration_discriminator2
            # duration_discriminator_type = getattr(hps.model, "duration_discriminator_type", "dur_disc_1")
            duration_discriminator_type = self.hparams.duration_discriminator_type
            print(f"Using duration discriminator {duration_discriminator_type} for VITS2")
            assert duration_discriminator_type in AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys(), f"duration_discriminator_type must be one of {list(AVAILABLE_DURATION_DISCRIMINATOR_TYPES.keys())}"
            #DurationDiscriminator = AVAILABLE_DURATION_DISCRIMINATOR_TYPES[duration_discriminator_type]

            if duration_discriminator_type == "dur_disc_1":
                self.net_dur_disc = DurationDiscriminatorV1(
                    self.hparams.hidden_channels,
                    self.hparams.hidden_channels,
                    3,
                    0.1,
                    gin_channels=self.hparams.gin_channels if self.hparams.num_speakers != 0 else 0,
                )
            elif duration_discriminator_type == "dur_disc_2":
                self.net_dur_disc = DurationDiscriminatorV2(
                    self.hparams.hidden_channels,
                    self.hparams.hidden_channels,
                    3,
                    0.1,
                    gin_channels=self.hparams.gin_channels if self.hparams.num_speakers != 0 else 0,
                )
        else:
            print("NOT using any duration discriminator like VITS1")
            self.net_dur_disc = None

        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

        # State kept between training optimizers
        self.x_mask = None
        self.hidden_x = None
        self.logw = None
        self.logw_ = None
        self._y = None
        self._y_hat = None
        self._y_hat_mb = None

    def _load_datasets(
        self,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ):
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset, max_phoneme_ids=max_phoneme_ids
        )
        valid_set_size = int(len(full_dataset) * validation_split)
        train_set_size = len(full_dataset) - valid_set_size - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def forward(self, text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )

        return audio

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)

        if optimizer_idx == 1:
            return self.training_step_d(batch)
        if optimizer_idx == 2:
            return self.training_step_dur(batch)

    def training_step_g(self, batch: Batch):
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )
        # VITS2:
        if self.model_g.use_noise_scaled_mas:
            current_mas_noise_scale = self.model_g.mas_noise_scale_initial - self.model_g.noise_scale_delta * self.global_step
            self.model_g.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        (
            y_hat,
            y_hat_mb,
            l_length,
            _attn,
            ids_slice,
            x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (hidden_x, logw, logw_)
        ) = self.model_g(x, x_lengths, spec, spec_lengths, speaker_ids)
        self.x_mask = x_mask
        self.hidden_x = hidden_x
        self.logw = logw
        self.logw_ = logw_
        self._y_hat = y_hat
        self._y_hat_mb = y_hat_mb
        if self.hparams.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
        y_mel = slice_segments(
            mel,
            ids_slice,
            self.hparams.segment_size // self.hparams.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.hparams.filter_length,
            self.hparams.mel_channels,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.mel_fmin,
            self.hparams.mel_fmax,
        )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Save for training_step_d
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)
        if self.net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(hidden_x, x_mask, logw_, logw)
        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            if self.hparams.mb_istft_vits:
                pqmf = PQMF(y)
                y_mb = pqmf.analysis(y)
                loss_subband = subband_stft_loss(
                    self.hparams.fft_sizes, self.hparams.hop_sizes, self.hparams.win_lengths,
                    y_mb, y_hat_mb
                )
            else:
                loss_subband = torch.tensor(0.0)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
            if self.net_dur_disc is not None:
                loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                loss_gen_all += loss_dur_gen
            self.log("loss_gen_all", loss_gen_all)

            return loss_gen_all

    def training_step_d(self, batch: Batch):
        # From training_step_g
        y = self._y
        y_hat = self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            # Discriminator
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            self.log("loss_disc_all", loss_disc_all)

            return loss_disc_all

    def training_step_dur(self, batch: Batch):
        hidden_x = self.hidden_x
        x_mask = self.x_mask
        logw_ = self.logw_
        logw = self.logw
        if self.net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = self.net_dur_disc(
                hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
            )  # logw is predicted duration, logw_ is real duration
            with autocast(self.device.type, enabled=False):
                loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                loss_dur_disc_all = loss_dur_disc
                self.log("loss_dur_disc_all", loss_dur_disc_all)

                return loss_dur_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch) + self.training_step_dur(batch)
        self.log("val_loss", val_loss)
        print(f"Epoch: {self.current_epoch}. Steps: {self.global_step}. Validation loss: {val_loss}")
        return val_loss

    def on_validation_end(self) -> None:
        # Generate audio examples after validation, but not during sanity check
        if not self.trainer.sanity_checking:
            for utt_idx, test_utt in enumerate(self._test_dataset):
                text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
                text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
                scales = [1.0, 1.0, 1.0]
                sid = (
                    test_utt.speaker_id.to(self.device)
                    if test_utt.speaker_id is not None
                    else None
                )
                test_audio = self(text, text_lengths, scales, sid=sid).detach()

                # Scale to make louder in [-1, 1]
                test_audio = test_audio * (1.0 / max(0.01, abs(test_audio).max()))

                tag = test_utt.text or str(utt_idx)
                self.logger.experiment.add_audio(
                    tag,
                    test_audio,
                    self.global_step,
                    sample_rate=self.hparams.sample_rate
                )

            return super().on_validation_end()

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.model_g.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
            torch.optim.AdamW(
                self.model_d.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ),
        ]
        if self.net_dur_disc is not None:
            optimizers.append(
                torch.optim.AdamW(
                    self.net_dur_disc.parameters(),
                    lr=self.hparams.learning_rate,
                    betas=self.hparams.betas,
                    eps=self.hparams.eps,
                )
            )

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[0], gamma=self.hparams.lr_decay
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers[1], gamma=self.hparams.lr_decay
            ),
        ]
        if self.net_dur_disc is not None:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizers[2], gamma=self.hparams.lr_decay
                )
            )
        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-test-examples", type=int, default=5)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        #
        parser.add_argument("--use_mel_posterior_encoder", type=bool, default=True)
        parser.add_argument("--use_transformer_flows", type=bool, default=True)
        parser.add_argument("--transformer_flow_type", type=str, default="pre_conv2")
        parser.add_argument("--use_spk_conditioned_encoder", type=bool, default=False)
        parser.add_argument("--use_noise_scaled_mas", type=bool, default=True)
        parser.add_argument("--use_duration_discriminator", type=bool, default=True)
        parser.add_argument("--duration_discriminator_type", type=str, default="dur_disc_2")
        parser.add_argument("--mb_istft_vits", type=bool, default=False)
        parser.add_argument("--ms_istft_vits", type=bool, default=False)
        parser.add_argument("--istft_vits", type=bool, default=True)
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        #
        return parent_parser
