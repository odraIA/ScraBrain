| model | model_display | run_dir | seed | train_pct | retrieval_size | top_k | balanced_top_k_accuracy | top_k_accuracy | chance_accuracy | balanced_accuracy_x_chance | accuracy_x_chance | n_samples | n_skipped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_init | Arquitectura aleatoria | logs/word_classification_weissbart_eeg/20260626_130140/random_init | 42 | 1.000000 | 50 | 10 | 0.195652 | 0.143293 | 0.200000 | 0.978261 | 0.716463 | 4264 | 4836 |
| random_init | Arquitectura aleatoria | logs/word_classification_weissbart_eeg/20260626_130140/random_init | 42 | 1.000000 | 250 | 10 | 0.044872 | 0.023715 | 0.040000 | 1.121795 | 0.592885 | 6578 | 2522 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_weissbart_eeg/20260626_130140/eeg_from_scratch | 42 | 1.000000 | 50 | 10 | 0.433653 | 0.504925 | 0.200000 | 2.168264 | 2.524625 | 4264 | 4836 |
| eeg_from_scratch | Checkpoint EEG desde cero | logs/word_classification_weissbart_eeg/20260626_130140/eeg_from_scratch | 42 | 1.000000 | 250 | 10 | 0.171253 | 0.283673 | 0.040000 | 4.281316 | 7.091821 | 6578 | 2522 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_weissbart_eeg/20260626_130140/eeg_pretrained | 42 | 1.000000 | 50 | 10 | 0.479673 | 0.531660 | 0.200000 | 2.398363 | 2.658302 | 4264 | 4836 |
| eeg_pretrained | Checkpoint EEG preentrenado | logs/word_classification_weissbart_eeg/20260626_130140/eeg_pretrained | 42 | 1.000000 | 250 | 10 | 0.192319 | 0.307996 | 0.040000 | 4.807965 | 7.699909 | 6578 | 2522 |
