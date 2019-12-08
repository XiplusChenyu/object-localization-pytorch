from Settings import Settings
from ReplayMemory import Transition
import torch

criterion = torch.nn.MSELoss().cuda() if Settings.cuda else nn.MSELoss()


def use_cuda(tensor, cuda=Settings.cuda):
    if cuda:
        tensor = tensor.cuda()
    return tensor


def optimize_model(policy_model, target_model, memory, optimizer):
    # target_model
    """
    Takes a replay memory, use this memory for model training,
    for one batch
    """

    if len(memory) < Settings.batch_size:
        return -1
    transitions = memory.sample(Settings.batch_size)
    batch = Transition(*zip(*transitions))
    # zip batches as a big named tuple

    non_finals = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_finals = use_cuda(non_finals)

    next_states = [s for s in batch.next_state if s is not None]
    if not next_states:
        return

    non_final_next_states = torch.cat(next_states).type(torch.float)
    non_final_next_states = use_cuda(non_final_next_states)

    state_batch = torch.cat(batch.state).type(torch.float)
    state_batch = use_cuda(state_batch)

    action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1)
    action_batch = use_cuda(action_batch)

    reward_batch = torch.tensor(batch.reward, dtype=torch.float).view(-1, 1)
    reward_batch = use_cuda(reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_model(state_batch).gather(1, action_batch-1)  # action batch should - 1

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(Settings.batch_size)
    next_state_values = use_cuda(next_state_values)
    
    next_state_values[non_finals] = target_model(non_final_next_states).max(1)[0].detach()
    next_state_values = next_state_values.view(-1, 1)
    

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * Settings.gamma) + reward_batch

    # Compute  loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    return round(float(loss.data), 4)
