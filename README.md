# Legal Text Summarization

- [Legal Text Summarization](#legal-text-summarization)
  - [Clone the Project](#clone-the-project)
  - [Installing Dependencies](#installing-dependencies)
  - [Getting the BIGPATENT Dataset](#getting-the-bigpatent-dataset)
  - [Converting Pretrained PEGASUS to PEGASUS-Longformer](#converting-pretrained-pegasus-to-pegasus-longformer)
  - [Fine-Tuning (Training)](#fine-tuning-training)
  - [Finding the Best Learning Rate](#finding-the-best-learning-rate)
  - [Finding the Highest Supported Batch Size](#finding-the-highest-supported-batch-size)
  - [Evaluating Your Model Against PEGASUS](#evaluating-your-model-against-pegasus)
  - [Hand-Picked Examples](#hand-picked-examples)

## Clone the Project
You can clone the project by running `git clone https://github.com/nimashoghi/dlt-summarization.git`.

## Installing Dependencies
You must first install all the dependencies in this for this project. You can do this by running the following command (this is assuming you've already cloned the project): `pip install -r requirements.txt`

## Getting the BIGPATENT Dataset
To get the BIGPATENT dataset, `cd` into the `datasets` directory and run the `get-dataset.sh` script. This will download the BIGPATENT dataset and extract the train, test, and validation archives. You can also run these commands manually (but please make sure you have `gdown` installed from pip).
```bash
gdown "https://drive.google.com/uc?export=download&id=1J3mucMFTWrgAYa3LuBZoLRR3CzzYD3fa"
tar xvzf bigPatentData.tar.gz
cd bigPatentData
tar xzvf test.tar.gz
tar xzvf train.tar.gz
tar xzvf val.tar.gz
```
When training, please make sure the directory variable (`DATASET_DIR` in `summarization/data/big_patent.py`) is set to the directory for your extracted dataset.

## Converting Pretrained PEGASUS to PEGASUS-Longformer
To convert the pretrained `google/pegasus-big_patent` model to our PEGASUS-Longformer model, please use the `convert-pegasus-to-longformer.py` script. The arguments for this command are shown below. If you do not provide any arguments, it will output your new model to `./converted-models/longformer-pegasus/`.

You can give parameters to the script. Below is a list of all accepted parameters for `convert-pegasus-to-longformer.py`:
```
usage: convert-pegasus-to-longformer.py [-h] [--base_model BASE_MODEL] [--tokenizer_name_or_path TOKENIZER_NAME_OR_PATH] [--save_model_to SAVE_MODEL_TO] [--attention_window ATTENTION_WINDOW]
                                        [--max_pos MAX_POS] [--skip_create]

Convert Pegasus to Longformer-Pegasus. Replaces Pegasus encoder's SelfAttnetion with LongformerSelfAttention

optional arguments:
  -h, --help            show this help message and exit
  --base_model BASE_MODEL
                        The name or path of the base model you want to convert
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The name or path of the tokenizer
  --save_model_to SAVE_MODEL_TO
                        The path to save the converted model
  --attention_window ATTENTION_WINDOW
                        attention window size for longformer self attention (one sided)
  --max_pos MAX_POS     maximum encoder positions
  --skip_create         skip create long model
```

## Fine-Tuning (Training)
To fine-tune your new PEGASUS-Longformer model, run the following command:
```bash
python train.py --model longformer_pegasus --gpus -1 --accelerator ddp --batch_size 1 --limit_train_batches 256 --limit_val_batches 32 --val_check_interval 128 --accumulate_grad_batches 16 --max_epochs 160
```
You can change the paramters to your liking. Below is a list of all accepted parameters for `train.py`:
```
usage: train.py [-h] [--model MODEL_NAME] [--datamodule DATAMODULE_NAME] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]] [--default_root_dir DEFAULT_ROOT_DIR]
                [--gradient_clip_val GRADIENT_CLIP_VAL] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES] [--num_processes NUM_PROCESSES] [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]]
                [--tpu_cores TPU_CORES] [--log_gpu_memory LOG_GPU_MEMORY] [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM]
                [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]] [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS]
                [--max_steps MAX_STEPS] [--min_steps MIN_STEPS] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
                [--val_check_interval VAL_CHECK_INTERVAL] [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--log_every_n_steps LOG_EVERY_N_STEPS] [--accelerator ACCELERATOR]
                [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION] [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH] [--num_sanity_val_steps NUM_SANITY_VAL_STEPS]
                [--truncated_bptt_steps TRUNCATED_BPTT_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler [PROFILER]] [--benchmark [BENCHMARK]] [--deterministic [DETERMINISTIC]]
                [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]] [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]] [--terminate_on_nan [TERMINATE_ON_NAN]]
                [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL]
                [--distributed_backend DISTRIBUTED_BACKEND] [--automatic_optimization [AUTOMATIC_OPTIMIZATION]]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL_NAME    model name
  --datamodule DATAMODULE_NAME
                        datamodule name
  --logger [LOGGER]     Logger (or iterable collection of loggers) for experiment tracking.
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. Default: ``True``. .. warning:: Passing a ModelCheckpoint instance to this argument is deprecated since v1.1.0 and
                        will be unsupported from v1.3.0.
  --default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
  --gradient_clip_val GRADIENT_CLIP_VAL
                        0 means don't clip.
  --process_position PROCESS_POSITION
                        orders the progress bar when running multiple models on same machine.
  --num_nodes NUM_NODES
                        number of GPU nodes for distributed training.
  --num_processes NUM_PROCESSES
  --gpus GPUS           number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        If enabled and `gpus` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in "exclusive mode", such that only one process at a
                        time can access them.
  --tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
  --log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.
  --overfit_batches OVERFIT_BATCHES
                        Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0
  --track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs.
  --fast_dev_run [FAST_DEV_RUN]
                        runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the dict.
  --max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached.
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs
  --max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (None).
  --min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (None).
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (floats = percent, int = num_batches)
  --limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (floats = percent, int = num_batches)
  --limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (floats = percent, int = num_batches)
  --val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to check within a training epoch, use int to check every n steps (batches).
  --flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100 steps).
  --log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50 steps).
  --accelerator ACCELERATOR
                        Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also take in an accelerator object for custom hardware.
  --sync_batchnorm [SYNC_BATCHNORM]
                        Synchronize batch norm layers between process groups/whole world.
  --precision PRECISION
                        Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.
  --weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins.
  --weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use this if for whatever reason you need the checkpoints stored in a different place than
                        the logs written in `default_root_dir`. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/' Defaults to `default_root_dir`.
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches in all validation dataloaders. Default: 2
  --truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        Truncated back prop breaks performs backprop every k steps of much longer sequence.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        To resume training from a specific checkpoint pass in the path here. This can be a URL.
  --profiler [PROFILER]
                        To profile individual steps during training and assist in identifying bottlenecks. Passing bool value is deprecated in v1.1 and will be removed in v1.3.
  --benchmark [BENCHMARK]
                        If true enables cudnn.benchmark.
  --deterministic [DETERMINISTIC]
                        If true enables cudnn.deterministic.
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        Set to True to reload dataloaders every epoch.
  --auto_lr_find [AUTO_LR_FIND]
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize initial learning for faster convergence. trainer.tune() method will set the suggested
                        learning rate in self.lr or self.learning_rate in the LightningModule. To use a different key set a string instead of True with the key name.
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        Explicitly enables or disables sampler replacement. If not specified this will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for train sampler and
                        ``shuffle=False`` for val/test sampler. If you want to customize it, you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
  --terminate_on_nan [TERMINATE_ON_NAN]
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        If set to True, will `initially` run a batch size finder trying to find the largest batch size that fits into memory. The result will be stored in self.batch_size in the
                        LightningModule. Additionally, can be set to either `power` that estimates the batch size through a power search or `binsearch` that estimates the batch size through a binary
                        search.
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
  --amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or "apex")
  --amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...).
  --distributed_backend DISTRIBUTED_BACKEND
                        deprecated. Please use 'accelerator'
  --automatic_optimization [AUTOMATIC_OPTIMIZATION]
                        If False you are responsible for calling .backward, .step, zero_grad. Meant to be used with multiple optimizers by advanced users.
```

## Finding the Best Learning Rate
To find the best learning rate for a specific network, append the `--auto_lr_find` flag to your the training command from above. Then, the trainer will try progressively increasing the learning rate until it finds the optimal one.

## Finding the Highest Supported Batch Size
To find the highest batch size that will still fit in your GPU memory during training, append the `--auto_scale_batch_size binsearch` flag to your training command from above. This will use a binary search algorithm to find the most optimal batch size for training.

## Evaluating Your Model Against PEGASUS
To evaluate your model and compare it to PEGASUS results, run the `evaluate.py` script. See the example below:
```bash
python evaluate.py --longformer_pegasus_checkpoint ./checkpoints-bigpatent/checkpoints-best/epoch=78.ckpt --pegasus_pretrained_model google/pegasus-big_patent --random-seed 25
```

You may pass the following parameters to this script:
```
usage: evaluate.py [-h] [--longformer_pegasus_checkpoint LONGFORMER_PEGASUS_CHECKPOINT] [--pegasus_pretrained_model PEGASUS_PRETRAINED_MODEL] [--random_seed RANDOM_SEED] [--num_samples NUM_SAMPLES]
                   [--top_length_samples]

Evaluates Longformer-PEGASUS

optional arguments:
  -h, --help            show this help message and exit
  --longformer_pegasus_checkpoint LONGFORMER_PEGASUS_CHECKPOINT
                        The name or path of the base model you want to convert
  --pegasus_pretrained_model PEGASUS_PRETRAINED_MODEL
                        The name or path of the base model you want to convert
  --random_seed RANDOM_SEED
                        random seed for test selection (-1 = don't set random seed)
  --num_samples NUM_SAMPLES
                        number of test samples
  --top_length_samples  skip create long model
```

## Hand-Picked Examples
Please see the following document: [Hand-Picked Examples](examples.md)
