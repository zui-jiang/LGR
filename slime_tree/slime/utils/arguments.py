import argparse
import json
import logging
import os
from typing import Any

import yaml
from sglang_router.launch_router import RouterArgs
from transformers import AutoConfig

from slime.backends.sglang_utils.arguments import add_sglang_arguments
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )
            parser.add_argument(
                "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor"
            )
            parser.add_argument(
                "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--train-backend",
                type=str,
                choices=["megatron", "fsdp"],
                default="megatron",
                help="The backend for training.",
            )
            parser.add_argument(
                "--qkv-format",
                type=str,
                choices=["thd", "bshd"],
                default="thd",
                help="The qkv layout for Megatron backend.",
            )
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help="Whether to enable true-on-policy mode.",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--disable-weights-backuper",
                action="store_false",
                dest="enable_weights_backuper",
                help="Whether to disable weights backuper to save host memory.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN. 
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, *, evaluation=False) -> list[list[Sample]]`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `truncated`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )

            # RFT (Rejection Sampling Fine-Tuning)
            parser.add_argument(
                "--rft-reward-threshold",
                type=float,
                default=0.5,
                help=(
                    "Minimum reward threshold for RFT (Rejection Sampling Fine-Tuning). "
                    "Only samples with reward >= threshold will be kept for training. "
                    "Default: 0.5. For binary correctness (0 or 1), use 0.5 to keep only correct samples."
                ),
            )
            parser.add_argument(
                "--rft-min-correct-per-prompt",
                type=int,
                default=1,
                help=(
                    "Minimum number of correct samples required to keep a prompt group in RFT. "
                    "If a prompt has fewer correct samples than this, the entire group is discarded. "
                    "Default: 1 (keep if at least one correct)."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Interval for updating the weights",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-external",
                action="store_true",
                default=False,
                help="Use external SGLang instances instead of launching them inside the framework.",
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance the number of tokens between data parallel ranks with `karmarkar_karp` for verl. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument("--num-critic-only-steps", type=int, default=0, help="Number of critic only steps")
            parser.add_argument("--critic-load", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-save", type=str, default=None, help="The checkpoint for critic model.")
            parser.add_argument("--critic-lr", type=float, default=None, help="The lr for critic model")
            parser.add_argument(
                "--critic-lr-warmup-iters",
                type=int,
                default=0,
                help="number of iterations to linearly warmup for critic model.",
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean. (e.g., examples/Dr.GRPO/custom_reducer.py:get_pg_loss_reducer).",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--grpo-sample-filter",
                type=str,
                choices=["both", "positive", "negative"],
                default="both",
                help=(
                    "Filter GRPO samples by advantage sign. "
                    "'both': use all samples (default), "
                    "'positive': only update on samples with advantage > 0, "
                    "'negative': only update on samples with advantage < 0. "
                    "Filtering happens after advantage calculation."
                ),
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-use-sign-penalty",
                action="store_true",
                help=(
                    "Enable sign-dependent fixed penalty instead of proportional penalty. "
                    "Default (False): advantages = adv - coef * gopd_kl (penalty scales with KL). "
                    "With this flag (True): advantages = adv - fixed_penalty (same penalty for all tokens with same sign). "
                    "When enabled, --opd-kl-coef-positive/negative/zero become fixed penalty values instead of coefficients."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef-positive",
                type=float,
                default=None,
                help=(
                    "Without --opd-use-sign-penalty: KL penalty coefficient for positive reverse KL (student > teacher). "
                    "With --opd-use-sign-penalty: Fixed penalty value for positive tokens. "
                    "If not set, defaults to --opd-kl-coef. "
                    "Example: --opd-kl-coef-positive 1.5"
                ),
            )
            parser.add_argument(
                "--opd-kl-coef-negative",
                type=float,
                default=None,
                help=(
                    "Without --opd-use-sign-penalty: KL penalty coefficient for negative reverse KL (student < teacher). "
                    "With --opd-use-sign-penalty: Fixed penalty value for negative tokens. "
                    "If not set, defaults to --opd-kl-coef. "
                    "Example: --opd-kl-coef-negative 0.5"
                ),
            )
            parser.add_argument(
                "--opd-kl-coef-zero",
                type=float,
                default=None,
                help=(
                    "Without --opd-use-sign-penalty: KL penalty coefficient for zero reverse KL (student == teacher). "
                    "With --opd-use-sign-penalty: Fixed penalty value for exact matches. "
                    "If not set, defaults to --opd-kl-coef. "
                    "Exact matches are rare and usually don't need special handling. "
                    "Example: --opd-kl-coef-zero 0.0"
                ),
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            parser.add_argument(
                "--reopold",
                action="store_true",
                default=False,
                help="Enable REOPOLD (Relaxed On-Policy Distillation). Requires --use-opd.",
            )
            parser.add_argument(
                "--reopold-lambda",
                type=float,
                default=0.3,
                help="REOPOLD mixture coefficient λ. Clip threshold = log(λ/(1-λ)). Default 0.3.",
            )
            parser.add_argument(
                "--reopold-entropy-beta",
                type=float,
                default=0.2,
                help="REOPOLD entropy percentile β. Top-β fraction of tokens get gradient in Phase II. Default 0.2.",
            )
            parser.add_argument(
                "--reopold-switch-step",
                type=int,
                default=None,
                help="T_switch: Phase I→II transition step. Defaults to num_rollout // 3.",
            )
            parser.add_argument(
                "--opd-kl-filter-mode",
                type=str,
                choices=["threshold", "quantile"],
                default="threshold",
                help=(
                    "Mode for OPD KL-based loss mask filtering. "
                    "'threshold': Use fixed thresholds (--opd-kl-threshold-high/low). "
                    "'quantile': Use dynamic quantile-based thresholds (--opd-kl-quantile-high/low). "
                    "Default is 'threshold'."
                ),
            )
            parser.add_argument(
                "--opd-kl-threshold-high",
                type=float,
                default=None,
                help=(
                    "Upper threshold for KL filtering. Keeps tokens with KL <= threshold_high. "
                    "Used with --opd-kl-filter-mode threshold. Default is None (no upper limit)."
                ),
            )
            parser.add_argument(
                "--opd-kl-threshold-low",
                type=float,
                default=None,
                help=(
                    "Lower threshold for KL filtering. Keeps tokens with KL >= threshold_low. "
                    "Used with --opd-kl-filter-mode threshold. Default is None (no lower limit)."
                ),
            )
            parser.add_argument(
                "--opd-kl-quantile-high",
                type=float,
                default=None,
                help=(
                    "Upper quantile for KL filtering (e.g., 0.8 to filter top 20% high KL). "
                    "Used with --opd-kl-filter-mode quantile. Default is None."
                ),
            )
            parser.add_argument(
                "--opd-kl-quantile-low",
                type=float,
                default=None,
                help=(
                    "Lower quantile for KL filtering (e.g., 0.2 to filter bottom 20% low KL). "
                    "Used with --opd-kl-filter-mode quantile. Default is None."
                ),
            )
            parser.add_argument(
                "--opd-kl-filter-invert",
                action="store_true",
                help=(
                    "Invert KL filter logic: keep tokens outside thresholds instead of inside. "
                    "Default (False): keep threshold_low <= KL <= threshold_high. "
                    "With this flag (True): keep KL < threshold_low or KL > threshold_high."
                ),
            )

            # G-OPD (Generalized On-Policy Distillation) arguments
            parser.add_argument(
                "--opd-lambda",
                type=float,
                default=1.0,
                help=(
                    "G-OPD reward scaling factor (λ). "
                    "λ = 1.0: Standard OPD (default). "
                    "0 < λ < 1: Reward interpolation. "
                    "λ > 1: Reward extrapolation (ExOPD, paper recommends 1.25). "
                    "Reference: arXiv:2602.12125v1"
                ),
            )
            parser.add_argument(
                "--opd-filter-transform",
                type=str,
                choices=["identity", "exp"],
                default="identity",
                help=("Mode for OPD KL filter"),
            )
            parser.add_argument(
                "--opd-use-reward-correction",
                action="store_true",
                help=(
                    "Enable reward correction for strong-to-weak distillation. "
                    "Uses teacher base model as reference instead of student ref model. "
                    "Requires --teacher-base-model-name and extra inference cost."
                ),
            )
            parser.add_argument(
                "--opd-use-orm",
                action="store_true",
                help=(
                    "Enable Outcome Reward Model (ORM) for task-level rewards. "
                    "If disabled (default), pure distillation with zero task rewards. "
                    "If enabled, combines ORM rewards (--rm-type) with teacher distillation."
                ),
            )
            parser.add_argument(
                "--teacher-base-model-name",
                type=str,
                default=None,
                help=(
                    "Teacher's base model name (before RL) for reward correction. "
                    "Used with --opd-use-reward-correction. "
                    "Queried via same --rm-url with model parameter."
                ),
            )
            # TopK OPD arguments
            parser.add_argument(
                "--opd-use-topk",
                action="store_true",
                default=False,
                help=(
                    "Enable TopK mode for on-policy distillation. "
                    "Uses topk candidate tokens for more accurate KL estimation with weighted sum: "
                    "KL ≈ Σ_i p_student(token_i) * [log p_student(token_i) - log p_teacher(token_i)]. "
                    "This provides better KL approximation than single sampled token."
                ),
            )
            parser.add_argument(
                "--opd-topk-size",
                type=int,
                default=10,
                help=(
                    "Number of top-k tokens to use for TopK OPD mode. "
                    "Only used when --opd-use-topk is enabled. "
                    "Default is 10. Typical values: 5-20."
                ),
            )
            parser.add_argument(
                "--opd-teacher-topk-kl",
                action="store_true",
                default=False,
                help=(
                    "Enable teacher-topk driven KL distillation loss (computed during training forward pass). "
                    "Teacher's own top-k tokens define the token set; student log probs are gathered from "
                    "training logits with full gradient support and vocab-parallel awareness. "
                    "If the student's sampled token is not in the teacher's top-k, it replaces the last entry."
                ),
            )
            parser.add_argument(
                "--opd-teacher-topk-kl-type",
                type=str,
                default="backward",
                choices=["forward", "backward", "jsd"],
                help=(
                    "KL divergence type for teacher-topk KL loss. "
                    "'backward': KL(teacher || student) = Σ p_t * (log p_t - log p_s), student gets gradient. "
                    "'forward': KL(student || teacher) = Σ p_s * (log p_s - log p_t), student gets gradient. "
                    "'jsd': Jensen-Shannon divergence = 0.5*(KL(p_t||m) + KL(p_s||m)) where m=(p_t+p_s)/2. "
                    "Default is 'backward'."
                ),
            )
            parser.add_argument(
                "--opd-teacher-topk-kl-coef",
                type=float,
                default=1.0,
                help="Loss coefficient for teacher-topk KL distillation loss. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-teacher-topk-jsd-beta",
                type=float,
                default=0.5,
                help=(
                    "Mixture weight for JSD when --opd-teacher-topk-kl-type=jsd. "
                    "The mixture distribution is m = beta * p_student + (1-beta) * p_teacher. "
                    "JSD = (1-beta) * KL(p_t || m) + beta * KL(p_s || m). "
                    "Default is 0.5 (standard symmetric JSD). "
                    "Values < 0.5 put more weight on teacher side; > 0.5 on student side."
                ),
            )
            parser.add_argument(
                "--opd-teacher-topk-size",
                type=int,
                default=50,
                help=(
                    "Number of top-k logprobs to request from the teacher. "
                    "Used for teacher-topk KL loss (--opd-teacher-topk-kl), "
                    "teacher_logit_y collection, RC-OPD, and confidence reward (max_logp/entropy). Default is 50."
                ),
            )
            # Union TopK KL: use student top-k ∪ teacher top-k for KL loss and loss_mask filtering
            parser.add_argument(
                "--opd-union-topk-kl",
                action="store_true",
                default=False,
                help=(
                    "Enable union topk KL mode. Replaces teacher-only topk KL loss with "
                    "student top-k ∪ teacher top-k union set KL. Gradients flow through all "
                    "union tokens (up to 2k). No renormalization; uses raw log probs."
                ),
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter",
                action="store_true",
                default=False,
                help=(
                    "Enable loss_mask filtering based on union topk KL values. "
                    "Tokens with KL inside [low, high] are masked out to avoid "
                    "low-KL tokens diluting gradients. Requires --opd-union-topk-kl."
                ),
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter-mode",
                type=str,
                default="threshold",
                choices=["threshold", "quantile"],
                help="Filter mode: 'threshold' uses fixed values, 'quantile' uses batch-level percentiles.",
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter-threshold-low",
                type=float,
                default=None,
                help="KL lower bound. Tokens with KL >= low and KL <= high are filtered out.",
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter-threshold-high",
                type=float,
                default=None,
                help="KL upper bound. Tokens with KL >= low and KL <= high are filtered out.",
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter-quantile-low",
                type=float,
                default=None,
                help="Lower quantile (e.g. 0.2) for dynamic threshold computation within batch.",
            )
            parser.add_argument(
                "--opd-union-topk-kl-filter-quantile-high",
                type=float,
                default=None,
                help="Upper quantile (e.g. 0.8) for dynamic threshold computation within batch.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence",
                action="store_true",
                default=False,
                help="Enable teacher next-token confidence reward for union topk KL.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-coef",
                type=float,
                default=0.1,
                help="Coefficient for the confidence reward term.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-k",
                type=int,
                default=5,
                help="Number of student top-K candidates for confidence (can differ from union k).",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-entropy-threshold",
                type=float,
                default=1.0,
                help="Student entropy threshold; only high-entropy positions get confidence reward.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-model-name",
                type=str,
                default=None,
                help="Model name for tree-attention confidence queries. If not set, uses --teacher-model-name.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-urls",
                type=str,
                nargs="+",
                default=None,
                help="URL(s) for tree-attention confidence queries. If multiple, one is randomly chosen per request. If not set, uses --rm-url.",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-normalize-std",
                action="store_true",
                default=False,
                help="If set, normalize confidence by (x-mean)/std; otherwise just (x-mean).",
            )
            parser.add_argument(
                "--opd-union-topk-confidence-max-tree-tokens",
                type=int,
                default=0,
                help=(
                    "Max total tokens per tree-attention request for confidence queries. "
                    "0 = unlimited. When exceeded, split into multiple requests "
                    "sharing the same trunk (leverages SGLang prefix caching)."
                ),
            )
            parser.add_argument(
                "--opd-union-topk-confidence-alternate-steps",
                type=int,
                default=0,
                help=(
                    "Alternate between pure reverse-KL and pure confidence loss every N steps. "
                    "0 = no alternation (use combined loss). "
                    "When N>0: steps [0,N) use pure reverse-KL, steps [N,2N) use pure confidence, then cycle."
                ),
            )
            parser.add_argument(
                "--opd-union-topk-confidence-via-advantage",
                action="store_true",
                default=False,
                help=(
                    "Route union-topk confidence signal through the advantage path instead of "
                    "the direct union-topk KL loss. When enabled, confidence bonus is added to "
                    "advantages (so gradients only flow through the sampled token at each position, "
                    "matching standard OPD behavior), and confidence is disabled in compute_union_topk_kl."
                ),
            )
            parser.add_argument(
                "--opd-union-topk-confidence-via-advantage-coef",
                type=float,
                default=0.1,
                help=(
                    "Coefficient for the confidence bonus when applied via the advantage path "
                    "(--opd-union-topk-confidence-via-advantage). "
                    "Modifies advantages as: A_t' = A_t + coef * confidence_t."
                ),
            )
            parser.add_argument(
                "--opd-dualsample-truncate-by-teacher-logit-y",
                action="store_true",
                default=False,
                help=(
                    "Enable truncation of the KL penalty based on teacher's logit for sampled token y. "
                    "Truncates per-sample KL at the first position where the teacher becomes uncertain."
                ),
            )
            parser.add_argument(
                "--opd-dualsample-truncate-window-size",
                type=int,
                default=32,
                help="Lookahead window size for teacher_logit_y truncation. Default is 32.",
            )
            parser.add_argument(
                "--opd-dualsample-truncate-threshold",
                type=float,
                default=None,
                help=(
                    "Fixed threshold for teacher_logit_y truncation (lookahead mode). "
                    "Truncates at the first position where mean(t_logit_y[pos:pos+window]) < threshold. "
                    "If None, uses lookback rolling-mean mode instead."
                ),
            )
            parser.add_argument(
                "--opd-kl-prefix-mask-len",
                type=int,
                default=None,
                help=(
                    "Mask reverse-KL for the first N tokens of each response during OPD training. "
                    "Only tokens at position >= N (0-indexed) contribute to the KL penalty. "
                    "Example: --opd-kl-prefix-mask-len 9000 skips the first 9k response tokens."
                ),
            )
            parser.add_argument(
                "--opd-kl-decay",
                type=float,
                default=1.0,
                help=(
                    "Per-position exponential decay factor for the reverse-KL penalty. "
                    "At position t (0-indexed), the KL contribution is multiplied by decay^t. "
                    "decay=1.0 (default) means no decay. "
                    "decay=0.999 down-weights later tokens, focusing the KL signal on the beginning of the response. "
                    "Example: --opd-kl-decay 0.999"
                ),
            )
            parser.add_argument(
                "--opd-kl-future-gamma",
                type=float,
                default=1.0,
                help=(
                    "Future-weighted KL: replace per-token KL at position t with a gamma-discounted "
                    "weighted average of KL from position t to end of sequence. "
                    "kl_future[t] = sum_{s=t}^{T} gamma^(s-t)*kl[s] / sum_{s=t}^{T} gamma^(s-t). "
                    "gamma=1.0 (default) disables this feature (no change to per-token KL). "
                    "gamma<1.0 weights near-future tokens more heavily than distant ones. "
                    "Example: --opd-kl-future-gamma 0.99"
                ),
            )
            # RC-OPD arguments
            parser.add_argument(
                "--opd-use-rc",
                action="store_true",
                default=False,
                help=(
                    "Enable Rank-Consistency OPD (RC-OPD). "
                    "Classifies each response token as consistent or inconsistent based on whether "
                    "the ordering of (student token x, teacher token y) is the same in both distributions, "
                    "then applies per-class configurable loss. "
                    "Requires --opd-type=sglang and teacher top-k logprobs (--opd-teacher-topk-size)."
                ),
            )
            parser.add_argument(
                "--opd-rc-consistent-loss",
                type=str,
                choices=["opd_kl", "mse", "none"],
                default="none",
                help=(
                    "Loss type for tokens where student and teacher agree on the ranking of x vs y. "
                    "'opd_kl': reverse KL contribution = log_ps(x) - log_pt(x). "
                    "'mse': (ps(x)-pt(x))^2 + (ps(y)-pt(y))^2. "
                    "'none': no loss for this class. Default is 'none'."
                ),
            )
            parser.add_argument(
                "--opd-rc-inconsistent-loss",
                type=str,
                choices=["opd_kl", "mse", "none"],
                default="opd_kl",
                help=(
                    "Loss type for tokens where student and teacher disagree on the ranking of x vs y. "
                    "'opd_kl': reverse KL contribution = log_ps(x) - log_pt(x). "
                    "'mse': (ps(x)-pt(x))^2 + (ps(y)-pt(y))^2. "
                    "'none': no loss for this class. Default is 'opd_kl'."
                ),
            )
            parser.add_argument(
                "--opd-rc-consistent-coef",
                type=float,
                default=1.0,
                help="Loss coefficient for consistent tokens in RC-OPD. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-rc-inconsistent-coef",
                type=float,
                default=1.0,
                help="Loss coefficient for inconsistent tokens in RC-OPD. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-rc-sign-eps",
                type=float,
                default=1e-6,
                help=(
                    "Epsilon threshold for sign comparison in RC-OPD consistency check. "
                    "Log-prob differences with absolute value <= eps are treated as zero (neutral), "
                    "avoiding misclassification due to floating-point noise. Default is 1e-6."
                ),
            )
            parser.add_argument(
                "--opd-rc-mse-y-gamma",
                type=float,
                default=1.0,
                help=(
                    "Weight for the y-side term in RC-OPD MSE loss. "
                    "MSE loss = (ps(x)-pt(x))^2 + gamma*(ps(y)-pt(y))^2. "
                    "Set < 1.0 to down-weight the y term, 0.0 to disable it. Default is 1.0."
                ),
            )
            parser.add_argument(
                "--opd-confidence-reward-coef",
                type=float,
                default=0.0,
                help=(
                    "Coefficient for teacher confidence reward term added to advantages. "
                    "Modifies advantages as: A_t' = A_t + coef * confidence_t. "
                    "0.0 disables (default). "
                    "The confidence signal type is set by --opd-confidence-type."
                ),
            )
            parser.add_argument(
                "--opd-confidence-type",
                type=str,
                default="logpt_x",
                choices=[
                    "logpt_x",
                    "ppl",
                    "near-ppl",
                    "max_logp",
                    "entropy",
                    "action-entropy",
                    "action-max_logp",
                    "delta-entropy",
                    "delta-max_logp",
                    "future-ppl",
                    "future-action-maxlogp",
                ],
                help=(
                    "Type of teacher confidence signal used when --opd-confidence-reward-coef != 0. "
                    "'logpt_x': per-token log p_teacher(x_t) — teacher's log-prob of the student's "
                    "sampled token at each position (requires teacher_log_probs). "
                    "'ppl': per-token prefix perplexity of teacher on the response; at position t "
                    "computes -exp(-1/(t+1)*sum_{i<=t} log p_teacher(x_i)); lower prefix PPL = "
                    "teacher more confident up to this point = higher reward "
                    "(requires teacher_log_probs). "
                    "'near-ppl': same as 'ppl' but only uses the previous 128 tokens (sliding window) "
                    "instead of the full prefix (requires teacher_log_probs). "
                    "'max_logp': log-prob of teacher's top-1 token at each position — measures how "
                    "peaked the teacher distribution is (requires teacher_dist_topk_log_probs). "
                    "'entropy': negative entropy of teacher's distribution — higher means more "
                    "confident/concentrated teacher (requires teacher_dist_topk_log_probs). "
                    "'action-entropy': -H(pi_T|x_{<t+1}), teacher entropy after seeing x_t — "
                    "action reward version of 'entropy', shifted by one position "
                    "(requires teacher_dist_topk_log_probs). "
                    "'action-max_logp': max_logp[t+1], teacher top-1 confidence after seeing x_t — "
                    "action reward version of 'max_logp', shifted by one position "
                    "(requires teacher_dist_topk_log_probs). "
                    "'delta-entropy': H(pi_T|x_{<t}) - H(pi_T|x_{<t},x_t), i.e. entropy reduction "
                    "caused by x_t; positive when the student token pulls context toward the teacher's "
                    "familiar distribution, sharpening its next-step prediction "
                    "(requires teacher_dist_topk_log_probs). "
                    "'delta-max_logp': max_logp[t+1] - max_logp[t], increase in teacher top-1 "
                    "confidence caused by x_t; positive when x_t makes the teacher more peaked at "
                    "the next position (requires teacher_dist_topk_log_probs). "
                    "Default is 'logpt_x'."
                ),
            )
            parser.add_argument(
                "--opd-confidence-future-gamma",
                type=float,
                default=1.0,
                help=(
                    "Gamma for future-weighted confidence reward. "
                    "conf_reward[t] = sum_{s=t+1}^{T} gamma^(s-t-1) * raw[s]. "
                    "1.0 = no decay (sum all equally). 0.99 = effective window ~100 tokens. "
                    "Used with 'future-ppl' and 'future-action-maxlogp' confidence types."
                ),
            )
            parser.add_argument(
                "--opd-confidence-ppl-window",
                type=int,
                default=None,
                help=(
                    "Window size N for 'future-ppl' confidence type. "
                    "At position t, PPL is computed over tokens [t+1, t+N]. "
                    "None = use all remaining tokens (limited only by gamma decay)."
                ),
            )
            parser.add_argument(
                "--opd-confidence-clip-min",
                type=float,
                default=None,
                help=(
                    "Minimum value to clip the teacher confidence signal to, after normalization. "
                    "All confidence types have max=0; this controls the lower bound, e.g. "
                    "--opd-confidence-clip-min -10.0 prevents extreme penalties at very uncertain positions. "
                    "Default is None (no clipping)."
                ),
            )
            return parser

        def add_router_arguments(parser):
            parser.add_argument(
                "--use-slime-router",
                action="store_true",
                default=False,
                help="Whether to use SlimeRouter for text-based routing instead of SGLang token-based routing",
            )
            parser.add_argument(
                "--slime-router-middleware-paths",
                type=str,
                nargs="+",
                default="",
            )
            parser.add_argument(
                "--slime-router-timeout",
                type=float,
                default=None,
                help="Timeout for SlimeRouter HTTP requests in seconds.",
            )
            parser.add_argument(
                "--slime-router-max-connections",
                type=int,
                default=None,
                help="Max connections for SlimeRouter HTTP client.",
            )
            parser.add_argument(
                "--slime-router-health-check-failure-threshold",
                type=int,
                default=3,
                help="Number of consecutive failures before marking a worker as unhealthy.",
            )
            RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-no-resume", action="store_true", default=False)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether to always use train step as the step metric in wandb. "
                    "If set, we will always use the train steps for wandb logging, "
                    "otherwise, will use rollout step for most info other than train/*. "
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument.",
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Load the rollout data from this path for debugging. "
                    "The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. "
                    "When this is enabled, slime will not instantiate sglang servers."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--debug-rollout-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the rollout generation without training. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--debug-train-only",
                action="store_true",
                default=False,
                help=(
                    "Whether to only run the training without sglang servers. "
                    "This is useful for debugging the rollout generation function."
                ),
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-student-topk-size",
                type=int,
                default=0,
                help=(
                    "If > 0, compute and save student top-k tokens and log probs during training forward pass "
                    "into the debug train data dump (requires --save-debug-train-data or --dump-details). "
                    "Uses vocab-parallel-aware top-k to handle tensor parallelism correctly. Default is 0 (disabled)."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--rm-max-concurrent-requests",
                type=int,
                default=32,
                help=(
                    "Maximum number of concurrent requests to reward model service. "
                    "Limits parallelism to prevent overwhelming external RM services. "
                    "Applies to all async RMs including: remote_rm, custom RMs, OPD teacher models. "
                    "Default: 32. Reduce to 8-16 for large teacher models or low-throughput services."
                ),
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--eval-rm-type",
                type=str,
                default=None,
                help=(
                    "Type of the reward model used during evaluation. "
                    "If not set, will use --rm-type for both training and evaluation. "
                    "This is useful when training uses custom RM that returns dict (e.g., OPD with teacher logprobs) "
                    "but evaluation needs scalar rewards."
                ),
            )
            parser.add_argument(
                "--eval-custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function for evaluation. "
                    "If not set, will use --custom-rm-path for both training and evaluation. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--teacher-model-name",
                type=str,
                default=None,
                help=(
                    "Optional model name to pass to teacher model server (for multi-model SGLang server). "
                    "Used with --opd-type sglang to select specific teacher model in reward_func."
                ),
            )
            parser.add_argument(
                "--teacher-temperature",
                type=float,
                default=None,
                help=(
                    "Sampling temperature used when querying the teacher model for log-probs. "
                    "If not set, falls back to --rollout-temperature. "
                    "Overridden by --teacher-temperature-anneal-* when annealing is configured. "
                    "Example: --teacher-temperature 0.7"
                ),
            )
            parser.add_argument(
                "--teacher-temperature-anneal-start-step",
                type=int,
                default=None,
                help="Rollout step at which teacher temperature annealing begins.",
            )
            parser.add_argument(
                "--teacher-temperature-anneal-end-step",
                type=int,
                default=None,
                help=(
                    "Rollout step at which teacher temperature annealing ends. "
                    "After this step the temperature is fixed at --teacher-temperature-anneal-end-value."
                ),
            )
            parser.add_argument(
                "--teacher-temperature-anneal-start-value",
                type=float,
                default=None,
                help="Teacher temperature at the start of the annealing window.",
            )
            parser.add_argument(
                "--teacher-temperature-anneal-end-value",
                type=float,
                default=None,
                help="Teacher temperature at the end of the annealing window (and fixed thereafter).",
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            parser.add_argument(
                "--disable-rollout-trim-samples",
                action="store_true",
                default=False,
                help="disable trim samples in rollout buffer when converting samples to train data",
            )
            parser.add_argument(
                "--use-dynamic-global-batch-size",
                action="store_true",
                default=False,
                help="enable dynamic global batch size, disable trim samples in rollout buffer when converting samples to train data",
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_prefill_decode_disaggregation_arguments(parser):
            parser.add_argument(
                "--prefill-num-servers",
                type=int,
                default=None,
                help="Number of prefill servers for disaggregation.",
            )
            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-metric-checker-key",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-metric-checker-threshold",
                type=float,
                default=None,
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        def add_sglang_tp_size():
            temp_parser = argparse.ArgumentParser(add_help=False)
            temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
            temp_parser.add_argument("--sglang-pp-size", type=int, default=1)
            temp_parser.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
            temp_args, _ = temp_parser.parse_known_args()
            # Use sglang_pp_size if set (non-default), otherwise use sglang_pipeline_parallel_size
            pp_size = (
                temp_args.sglang_pp_size if temp_args.sglang_pp_size != 1 else temp_args.sglang_pipeline_parallel_size
            )
            sglang_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
            return sglang_tp_size

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_router_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_sglang_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_prefill_decode_disaggregation_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        parser.set_defaults(sglang_tensor_parallel_size=add_sglang_tp_size())
        return parser

    return add_slime_arguments


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    backend = parse_args_train_backend()
    if backend == "megatron":
        from slime.backends.megatron_utils.arguments import parse_args as megatron_parse_args
        from slime.backends.megatron_utils.arguments import set_default_megatron_args
        from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

        args = megatron_parse_args(extra_args_provider=add_slime_arguments)
        if args.hf_checkpoint and not args.debug_rollout_only:
            hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            hf_validate_args(args, hf_config)

        args.rank = 0
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
        args = set_default_megatron_args(args)
    else:
        logger.warning(
            "🚧 🚧 🚧 FSDP backend is being rewritten, please use Megatron backend for better stability. 🚧 🚧 🚧"
        )

        from slime.backends.fsdp_utils.arguments import load_fsdp_args

        args = load_fsdp_args(extra_args_provider=add_slime_arguments)
        args.rank = 0  # Primary process rank for wandb initialization
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    slime_validate_args(args)

    if backend == "megatron":
        megatron_validate_args(args)

        # always use varlen
        args.variable_seq_lengths = True
        if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
            logger.info(
                "--moe-token-dispatcher-type allgather does not support variable sequence length, "
                "please use alltoall dispatcher instead."
            )
            args.moe_token_dispatcher_type = "alltoall"

        if args.pipeline_model_parallel_size == 1:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
                "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
                "pipeline_model_parallel_size is 1."
            )

    sglang_validate_args(args)

    return args


def parse_args_train_backend():
    if os.environ.get("SLIME_BACKEND") is not None:
        raise Exception("`SLIME_BACKEND` is deprecated, please use --train-backend directly.")

    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)
    args_partial, _ = parser.parse_known_args()
    return args_partial.train_backend


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def slime_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )

        # Validate G-OPD specific arguments
        if args.opd_lambda < 0:
            raise ValueError(f"--opd-lambda must be >= 0, got {args.opd_lambda}")

        if args.opd_lambda != 1.0:
            if not args.opd_use_reward_correction:
                if args.kl_coef == 0:
                    raise ValueError(
                        f"G-OPD with --opd-lambda={args.opd_lambda} requires student's reference model. "
                        "Either set --kl-coef > 0 or use --opd-use-reward-correction with --teacher-base-model-name."
                    )
            else:
                if args.teacher_base_model_name is None:
                    raise ValueError("--opd-use-reward-correction enabled but --teacher-base-model-name not specified")

        if args.opd_use_orm and not args.rm_type and not args.custom_rm_path:
            raise ValueError(
                "--opd-use-orm enabled but no ORM configured. "
                "Set --rm-type (e.g., deepscaler/math/f1) or --custom-rm-path"
            )

        if args.opd_use_reward_correction and args.opd_lambda == 1.0:
            logger.warning(
                "[G-OPD] --opd-use-reward-correction has no effect when --opd-lambda=1.0. "
                "Set --opd-lambda > 1 (e.g., 1.25) for ExOPD with reward correction."
            )

        # Set default values for sign-dependent KL coefficients
        if args.opd_kl_coef_positive is None:
            args.opd_kl_coef_positive = args.opd_kl_coef
        if args.opd_kl_coef_negative is None:
            args.opd_kl_coef_negative = args.opd_kl_coef
        if args.opd_kl_coef_zero is None:
            args.opd_kl_coef_zero = args.opd_kl_coef

        # Log asymmetric configuration
        use_sign_penalty = getattr(args, "opd_use_sign_penalty", False)
        if args.opd_kl_coef_positive != args.opd_kl_coef_negative:
            if use_sign_penalty:
                logger.info(
                    f"[OPD] Using sign-dependent FIXED penalties: "
                    f"positive={args.opd_kl_coef_positive}, "
                    f"negative={args.opd_kl_coef_negative}, "
                    f"zero={args.opd_kl_coef_zero}"
                )
            else:
                logger.info(
                    f"[OPD] Using sign-dependent COEFFICIENTS: "
                    f"positive={args.opd_kl_coef_positive}, "
                    f"negative={args.opd_kl_coef_negative}, "
                    f"zero={args.opd_kl_coef_zero}"
                )
        elif use_sign_penalty:
            logger.info(f"[OPD] Using sign-dependent fixed penalty mode with uniform value: {args.opd_kl_coef}")

    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    # TODO: During loading, we need to set the start_rollout_id here.
    if args.megatron_to_hf_mode == "bridge":
        if args.load is None:
            args.load = args.ref_load or args.hf_checkpoint
        args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            args.start_rollout_id = 0

    # Load wandb run ID for resume if checkpoint exists
    if args.use_wandb and args.load and os.path.exists(args.load) and (not args.wandb_no_resume):
        wandb_id_file = os.path.join(args.load, "wandb_run_id.txt")
        if os.path.exists(wandb_id_file) and args.wandb_run_id is None:
            with open(wandb_id_file, "r") as f:
                args.wandb_run_id = f.read().strip()
                logger.info(f"Resuming wandb run: {args.wandb_run_id} from checkpoint")

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.use_critic = args.advantage_estimator == "ppo"
    if args.critic_num_gpus_per_node is None:
        args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    if args.critic_num_nodes is None:
        args.critic_num_nodes = args.actor_num_nodes
    if args.critic_load is None:
        args.critic_load = args.load
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and (not args.rollout_num_gpus):
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus != args.actor_num_gpus_per_node * args.actor_num_nodes:
            logger.info(
                f"rollout_num_gpus {args.rollout_num_gpus} != actor_num_gpus_per_node {args.actor_num_gpus_per_node} "
                f"* actor_num_nodes {args.actor_num_nodes}, overriding rollout_num_gpus to match actor_num_gpus_per_node * actor_num_nodes."
            )
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
            if args.use_critic:
                args.rollout_num_gpus += args.critic_num_gpus_per_node * args.critic_num_nodes

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    # Validate GRPO sample filter
    if args.grpo_sample_filter != "both":
        if args.advantage_estimator not in ["grpo", "gspo"]:
            logger.warning(
                f"--grpo-sample-filter is set to '{args.grpo_sample_filter}' "
                f"but advantage_estimator is '{args.advantage_estimator}'. "
                f"Sample filtering is primarily designed for GRPO/GSPO."
            )

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    assert not (
        args.prefill_num_servers is not None and args.rollout_external
    ), "prefill_num_servers cannot be set when rollout_external is set."

    if args.qkv_format == "bshd":
        assert args.train_backend == "megatron", "bshd format is only supported for megatron backend."
        assert (
            args.use_dynamic_batch_size is False
        ), "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")


def hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rope_theta", "rotary_base", equal),
    ]:
        if hasattr(hf_config, hf_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))
