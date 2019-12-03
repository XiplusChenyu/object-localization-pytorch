import torch


class Settings:
    path_dataset = "../VOC2012"
    path_figure = "../Figures/"
    class_name = "cat"
    class_index = 7
    dataset = "trainval"
    model_path = "../Model/"

    # cuda settings
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda

    # cuda = True  # for text
    scale = float(3) / 4

    # Reinforcement Learning Agent
    action_num = 6  # except the start point
    history_num = 4  # record last four actions
    eps_start = 0.9  # epsilon greedy start epsilon
    eps_end = 0.05
    eps_decay = 0.1
    gamma = 0.9

    # Reward
    terminal_reward = 3
    step_reward = 1
    iou_threshold = 0.9

    # Training
    target_update = 3 
    batch_size = 50
    max_step = 5





