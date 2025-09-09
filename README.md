# DextrAH on Isaac Lab

DextrAH is a high-performance hand-arm grasping policy. This codebase provides the machinery required to train such a policy in Isaac Lab starting with privileged RL training followed by online distillation that swaps the input space to camera data.

## Installation
1. [Install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) Isaac Sim, Isaac Lab following the local conda install route.

**Note**: After you clone the Isaac Lab repository and before installation, checkout the tag `v2.2.1` before installation (can also work with `v2.0.2` with minor code changes):
```bash
        cd <IsaacLab>
        git checkout v2.2.1
```
2. Install geometric fabrics from this [repo](https://github.com/NVlabs/FABRICS) within your new conda env

3. Install Dextrah for Isaac Lab within your new conda env
```bash
        curl -sSL https://install.python-poetry.org | python3 - --version 1.8.3
        git lfs clone git@github.com:NVlabs/DEXTRAH.git
        cd <DEXTRAH>
        poetry install
        or
        python -m pip install -e .
```

## DextrAH Privileged FGP Teacher Training
1. Single-GPU training

**Note**: set `num_gpus_per_node` to the number of GPUs available, often 1. set `num_nodes` to number of training nodes (1 if running locally)

**Note**: `env.use_cuda_graph=True` uses a cuda graph capture of fabrics and makes training faster. It may lead to cuda memory issues in some cases.
```bash
        cd <DEXTRAH>/dextrah_lab/rl_games
        python -m torch.distributed.run --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node>\
          train.py \
            --headless \
            --task=Dextrah-Kuka-Allegro \
            --seed -1 \
            --distributed \
            --num_envs 4096 \
            agent.params.config.minibatch_size=16384 \
            agent.params.config.central_value_config.minibatch_size=16384 \
            agent.params.config.learning_rate=0.0001 \
            agent.params.config.horizon_length=16 \
            agent.params.config.mini_epochs=4 \
            agent.params.config.multi_gpu=True \
            agent.wandb_activate=False \
            env.success_for_adr=0.4 \
            env.objects_dir=visdex_objects \
            env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
            env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
            env.max_pose_angle=45.0 \
            env.use_cuda_graph=True
```
## DextrAH Camera-based FGP Student Distillation
**Note**: Before starting the student training, you also need to download the visual texture data (textures.zip) and place its contents inside `dextrah_lab/assets` directory. Download the assets from [link](https://huggingface.co/datasets/nvidia/dextrah_textures/blob/main/textures.zip) and unzip its contents into the assets folder.

1. Training

**Note**: If you want to train with additional data augmentation, you can pass the `--data_aug` flag, but this is often unnecessary.
```bash
        cd <DEXTRAH>/dextrah_lab/distillation
        # NOTE: in general we should try to use a perfect square number of tiles
        python -m torch.distributed.run --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node> \
          run_distillation.py \
            --headless
            --distributed \
            --task=Dextrah-Kuka-Allegro \
            --num_envs 256 env.distillation=True \
            --enable_cameras env.simulate_stereo=True \
            --teacher <path_to_teacher>  \
            env.img_aug_type="rgb" \
            env.aux_coeff=10. \
            env.objects_dir="visdex_objects" \
            env.max_pose_angle=45.0 \
            env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
            env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
            env.use_cuda_graph=True
```

2. Single-GPU evaluation
To eval (i.e. play) a trained student policy, run the following command:
```bash
        cd <DEXTRAH>/dextrah_lab/distillation
        python eval.py \
        --task=Dextrah-Kuka-Allegro \
        --num_envs 32 \
        --enable_cameras \
        --checkpoint <path_to_checkpoint> \
        --num_episodes 10 \
        env.distillation=True \
        env.simulate_stereo=True \
        env.img_aug_type="rgb" \
        env.objects_dir="visdex_objects" \
        env.max_pose_angle=45.0 \
        env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
        env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
        env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
        env.use_cuda_graph=True
```

The eval script also provide functions to record data. Passing the following
extra args for data recording.
```bash
        --record_data \
        --max_records_per_file 100 \
        --create_video
```
**Note:** By default, most of the randomization are turned off for data recording.

**Note:** The create video arg will create videos for the recorded data for easy data inspection.
However, it will slow down the process. It's recommended to only use it for debugging.

## Notes
One can update dependences in deps.txt file, remove pyproject.toml and poetry.lock files, and regenerate them
```bash
    cd <DEXTRAH>
    rm pyproject.toml poetry.lock
    poetry init --name "dextrah_lab" --no-interaction
    xargs poetry add < deps.txt
    poetry install
```
The `dextrah_lab/deployment_scripts` directory contains several reference scripts to show how to deploy the trained FGP, fabric controller, state machine, and camera calibration. These will not run out of the box because they depend on camera, PD controller, and robot driver ROS 2 nodes to be running, which are not included. Specifically, these are of interest
```bash
    camera_calibration.py
    camera_transform_publisher.py
    kuka_allegro_fabric.py
    kuka_allegro_state_machine.py
    kuka_allegro_stereo_fgp.py
```
