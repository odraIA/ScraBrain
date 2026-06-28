| model | model_display | run_dir | seed | train_pct | retrieval_size | top_k | balanced_top_k_accuracy | top_k_accuracy | chance_accuracy | balanced_accuracy_x_chance | accuracy_x_chance | n_samples | n_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | Arquitectura aleatoria | logs/word_classification_alice_eeg/20260627_123237/random_init | 42 | 1.000000 | 50 | 10 | 0.186392 | 0.162665 | 0.200000 | 0.931958 | 0.813325 | 4623 | 4727 |
| random_init | Arquitectura aleatoria | logs/word_classification_alice_eeg/20260627_123237/random_init | 42 | 1.000000 | 250 | 10 | 0.038482 | 0.043692 | 0.040000 | 0.962053 | 1.092312 | 7713 | 1637 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_alice_eeg/20260627_123237/eeg_from_scratch | 42 | 1.000000 | 50 | 10 | 0.826127 | 0.788882 | 0.200000 | 4.130635 | 3.944408 | 4623 | 4727 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_alice_eeg/20260627_123237/eeg_from_scratch | 42 | 1.000000 | 250 | 10 | 0.784040 | 0.758719 | 0.040000 | 19.601009 | 18.967976 | 7713 | 1637 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_alice_eeg/20260627_123237/eeg_pretrained | 42 | 1.000000 | 50 | 10 | 0.887186 | 0.787800 | 0.200000 | 4.435931 | 3.939001 | 4623 | 4727 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_alice_eeg/20260627_123237/eeg_pretrained | 42 | 1.000000 | 250 | 10 | 0.811526 | 0.763386 | 0.040000 | 20.288154 | 19.084662 | 7713 | 1637 |
