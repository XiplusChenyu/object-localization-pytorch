import torch


class Settings:
    path_dataset = "../VOC2012"
    path_figure = "../Figures/"
    dataset = "trainval"
    model_path = "../Model/"

    # cuda settings
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda

    # cuda = True  # for text
    scale = float(4) / 5

    # Reinforcement Learning Agent
    action_num = 6  # except the start point
    history_num = 4  # record last four actions
    eps_start = 0.9  # epsilon greedy start epsilon
    gamma = 0.99

    # Reward
    terminal_reward = 3
    step_reward = 1
    iou_threshold = 0.6

    # Training
    batch_size = 64
    max_step = 5





