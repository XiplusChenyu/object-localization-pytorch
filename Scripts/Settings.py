import torch


class Settings:
    path_dataset = "../VOC2012"
    path_figure = "../Figures/"
    class_name = "cat"
    class_index = 7
    dataset = "trainval"
    model_path = "../Model/model.pt"

    # cuda settings
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda

    # cuda = True  # for text
    scale = float(3) / 4

    # Reinforcement Learning Agent
    action_num = 6  # except the start point
    history_num = 4  # record last four actions
    eps_start = 0.8  # epsilon greedy start epsilon
    eps_end = 0.05
    eps_decay = 200
    gamma = 0.999

    # Reward
    terminal_reward = 3
    step_reward = 1
    iou_threshold = 0.5

    # Training
    target_update = 10
    batch_size = 50





