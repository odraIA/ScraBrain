"""Run continuous EEG training with the original MEG-XL model.

Only the EEG DataModule is replaced so it can preserve physical-run continuity
and provide target intervals for listeningcovert. The shared trainer imports the
same ``CrissCrossTransformerModule`` used by MEG-XL; no alternative attention or
EEG-specific transformer is installed here.
"""

from __future__ import annotations

import brainstorm.data.eeg_multi_datamodule as legacy_datamodule
from brainstorm.data.eeg_continuous_masked_datamodule import MultiEEGDataModule


legacy_datamodule.MultiEEGDataModule = MultiEEGDataModule

from brainstorm.train_criss_cross_eeg_multi import main  # noqa: E402


if __name__ == "__main__":
    main()
