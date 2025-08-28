# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 29
# TIME_LIMIT = 59
TIME_LIMIT = 360

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/mdpo/outputs"

BASE_RUN_NAME = f"debug"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

NODES = 1
GPN = 1
# NODES = 1
# GPN = 4
# NODES = 4
# GPN = 4

run_name = f"mdpo_N{NODES}n{NODES*GPN}"

ACCEL_CONFIG="recipes/accelerate_configs/zero2.yaml"

ACCEL_PREAMBLE=f"accelerate launch \
    --config_file {ACCEL_CONFIG} \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    --num_machines $SLURM_NNODES \
    --num_processes {NODES*GPN} \
"""

# SYS_PROMPT = r"Let's think step by step and output the final answer within \\boxed{}."
# --system_prompt "{SYS_PROMPT}" \
# instead using the system_prompt_type flag to utilize an internally defined copy of this since
# the string nesting isnt cooperating

# Cfgs
exp_list = [
    [f"""\
{ACCEL_PREAMBLE} \
src/open_r1/mdpo.py \
--config recipes/LLaDA-Instruct/mdpo/config_demo.yaml \
--dataset_train_split train \
--num_train_epochs 1 \
--dataset_name open-r1/OpenR1-Math-220k \
--save_strategy "epoch" \
--output_dir checkpoints/LLaDA-8B-Instruct-MDPO-numina-adv-128st-8sample_temp0.4_{NODES*GPN}gpus \
--num_generations {NODES*GPN} \
--learning_rate 7e-7 \
--gradient_accumulation_steps 16 \
--temperature 0.4 \
--beta 0.02 \
--block_length 512 \
--max_completion_length 512 \
--sample_train_steps 8 \
--max_prompt_length 320 \
--diffusion_steps 128 \
--remask_strategy random \
--num_train_samples 3500 \
--system_prompt_type step_by_step \
--incremental_training true \
--mixture_data true \
""", run_name]
]


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        run_name,
    ) = exp

    # put together the actual "train.py" command
    custom_invocation = f"{script}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --wandb_offline={WANDB_OFFLINE} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --qos={QOS} \
        --bank={BANK} \
        --minutes={TIME_LIMIT} \
        --nodes={NODES} \
        --gpus_per_node={GPN} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation} --output-dir={BASE_OUT_DIR}/{BASE_RUN_NAME}/{run_name}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        print(command)

print(f"Total launches: {total_launches}")
