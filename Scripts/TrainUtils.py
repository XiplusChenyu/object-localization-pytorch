from Settings import Settings
from ReplayMemory import Transition
import numpy as np
import torch


def use_cuda(tensor, cuda=Settings.cuda):
    if cuda:
        tensor = tensor.cuda()
    return tensor


def optimize_model(model, memory):
    """
    Takes a replay memory, use this memory for model training
    """
    if len(memory) < Settings.batch_size:
        return
    transitions = memory.sample(Settings.batch_size)
    batch = Transition(*zip(*transitions))
    # zip batches as a big named tuple

    non_finals = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8)
    non_finals = use_cuda(non_finals)

    with torch.no_grad():
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.tensor(next_states, dtype=torch.float)
        non_final_next_states = use_cuda(non_final_next_states)

    state_batch = Variable(torch.cat(batch.state)).type(torch.FloatTensor)
    action_batch = Variable(torch.LongTensor(batch.action).view(-1, 1)).type(torch.LongTensor)
    reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1, 1)).type(torch.FloatTensor)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE, 1).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute  loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()