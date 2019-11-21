import torch


class Settings:
    path_dataset = "../VOC2012"
    path_figure = "../Figures/"
    class_name = "cat"
    class_index = 7
    dataset = "trainval"

    # cuda settings
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda

    # cuda = True  # for text
    scale = float(3) / 4

    # Reinforcement Learning Agent
    action_num = 6  # except the start point
    history_num = 4  # record last four actions
    epsilon = 0.8  # epsilon greedy start epsilon
    gamma = 0.999

    # Reward
    terminal_reward = 3
    step_reward = 1

    # Training
    batch_size = 100




