# DextrAH on Isaac Lab

DextrAH is a high-performance hand-arm grasping policy. This codebase provides the machinery required to train such a policy in Isaac Lab starting with privileged RL training followed by online distillation that swaps the input space to camera data.

## Installation without docker
1. [Install](https://isaac-sim.github.io/IsaacLab/main/index.html) Isaac Lab and its dependencies.
**Note**: After you clone the Isaac Lab repository and before installation, check out the tag `v2.0.2` before installation:
```bash
        git fetch origin
        git checkout v2.0.2
```
2. Install geometric fabrics.
**Note**: Needs to checkout the branch `develop` for everything to work properly. This will be more formalized once we have a more stabilized main.
```bash
        git clone https://gitlab-master.nvidia.com/srl/fabrics-sim
        cd <fabrics-sim root>
        git checkout develop
        pip install .
        # If isaac sim was installed with pip or conda
        ./urdfpy_patch.sh
        # If isaac sim already installed within official docker container
        ./urdfpy_patch.sh --docker
```

3. Install Dextrah for Isaac Lab
```bash
        git clone https://gitlab-master.nvidia.com/kvanwyk/dextrah-lab-internal.git
        cd <dextrah-lab-internal>
        pip install -e .
```

## Installation with building Isaac Lab image and using docker
1. Download Isaac Lab from public repo and check out tag

        git clone https://github.com/isaac-sim/IsaacLab.git
        cd IsaacLab
        git checkout v2.0.2

2. Build base Isaac Lab docker image. First, [setup docker and log into nvcr.io](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#obtaining-the-isaac-sim-container). Then, build the base Isaac Lab image and shut down its container
```bash
        cd IsaacLab
        ./docker/container.py start
        ./docker/container.py stop
```
3. Download Dextrah
```bash
        git clone https://gitlab-master.nvidia.com/kvanwyk/dextrah-lab-internal.git
```
4. Spin up a new container that targets the base Isaac Lab image along with various env variables
```bash
        cd <dextrah-lab-internal/dextrah_lab/docker>
        docker run -it --env-file .env --entrypoint bash --gpus all --network host isaac-lab-base
```
5. In a separate terminal, commit the running container to a new image, isaac-lab-dextrah with v1 tag
```bash
        docker commit <container_id> isaac-lab-dextrah:v1
```
6. Kill the running container
```bash
        docker kill <container_id>
```
7. Target the new image along with binding in fabrics and dextrah repos:
```bash
        docker run -it --entrypoint bash -v /local_path_to_dextrah/dextrah-lab-internal:/workspace/dextrah-lab-internal -v /local_path_to_fabrics/fabrics-sim:/workspace/fabrics-sim --gpus all --network host isaac-lab-dextrah:v1
```
8. Install fabrics inside container
```bash
        cd /workspace/fabrics-sim
        /isaac-sim/python.sh -m pip install .
        ./urdfpy_patch --docker
```
9. Install dextrah inside container in editable mode
```bash
        cd /workspace/dextrah-lab-internal
        /isaac-sim/python.sh -m pip install -e .
```
10. Train Dextrah using same commands below inside the container except use `/isaac-sim/python.sh` instead of `python`. You may also consider to save a commit before existing the container to preserve all the installation.
```bash
        docker commit <container_id> isaac-lab-dextrah-installed
```
## DextrAH Privileged FGP Teacher Training
1. Single-GPU training
```bash
        cd <dextrah_lab root>/dextrah_lab/rl_games
        python train.py \
            --headless \
            --task=Dextrah-Kuka-Allegro \
            --seed -1 \
            --num_envs 4096 \
            agent.params.config.minibatch_size=16384 \
            agent.params.config.central_value_config.minibatch_size=16384 \
            agent.params.config.learning_rate=0.0001 \
            agent.params.config.horizon_length=16 \
            agent.params.config.mini_epochs=4 \
            agent.wandb_activate=False \
            env.success_for_adr=0.4 \
            env.objects_dir=visdex_objects \
            env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
            env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]" \
            env.max_pose_angle=45.0
```
2. Multi-GPU training (4 GPUs, 1 node)
```bash
        cd <dextrah_lab root>/dextrah_lab/rl_games
        python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
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
            env.max_pose_angle=45.0
```
## DextrAH Camera-based FGP Student Distillation
**Note**: Before starting the student training, you also need to prepare the `dextrah_lab/assets` folder. Download the assets from this [link](https://drive.google.com/drive/folders/18P9GOxtsotG8UR-dxqHHl5ZIm9lf3uZJ) and unzip it to the corresponding assets folder. You may have to request for permission to access the data. Note that all the objects inside the resulting `distilation_assets` folder should be moved out and placed directly in the `dextrah_lab/assets` folder.

1. Training
> The logger is default to WANDB and you may need to update the [entity](https://gitlab-master.nvidia.com/kvanwyk/dextrah-lab-internal/-/blame/main/dextrah_lab/tasks/dextrah_kuka_allegro/agents/rl_games_ppo_lstm_cfg.yaml?ref_type=heads#L129) for proper access. If you want to train with data augmentation, you can pass the `--data_aug` flag.
```bash
        cd <dextrah_lab root>/dextrah_lab/distillation
        # NOTE: in general we should try to use a perfect square number of tiles
        python -m torch.distributed.run --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node> \
          run_distillation.py \
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
            env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]"
```

2. Single-GPU evaluation
To eval (i.e. play) a trained student policy, run the following command:
```bash
        python eval.py \
        --task=Dextrah-Kuka-Allegro \
        --num_envs 32 \
        --enable_cameras \
        --checkpoint REPLAY/WITH/PATH/TO/CHECKPOINT \
        --num_episodes 10 \
        env.distillation=True \
        env.simulate_stereo=True \
        env.img_aug_type="rgb" \
        env.objects_dir="visdex_objects" \
        env.max_pose_angle=45.0 \
        env.adr_custom_cfg_dict.fabric_damping.gain="[10.0, 20.0]" \
        env.adr_custom_cfg_dict.reward_weights.finger_curl_reg="[-0.01, -0.01]" \
        env.adr_custom_cfg_dict.reward_weights.lift_weight="[5.0, 0.0]"
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
