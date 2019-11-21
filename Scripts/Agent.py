from Settings import Settings
from Models import get_image_feature
import random
import torch


class Agent:
    """
    An agent class mock the agent movement and agent status
    """
    def __init__(self, image):
        self.image = image[:]
        self.sub_image = self.image[:]
        self.boundary = {
            "xmin": 0,
            "ymin": 0,
            "xmax": image.shape[1],
            "ymax": image.shape[0]
        }
        self.actionMap = {
            0: "START",
            1: "up-left",
            2: "up-right",
            3: "down-left",
            4: "down-right",
            5: "center",
            6: "END"
        }
        self.history = torch.zeros((Settings.history_num, Settings.action_num))
        if Settings.cuda:
            self.history = self.history.cuda()

    def hierarchical_move(self, action, scale=Settings.scale):
        if action == 1:
            self.boundary["xmax"] *= scale
            self.boundary["ymax"] *= scale
        elif action == 2:
            x_width = self.boundary["xmax"] - self.boundary["xmin"]
            self.boundary["ymax"] *= scale
            self.boundary["xmin"] = self.boundary["xmax"] - scale * x_width
        elif action == 3:
            y_width = self.boundary["ymax"] - self.boundary["ymin"]
            self.boundary["xmax"] *= scale
            self.boundary["ymin"] = self.boundary["ymax"] - scale * y_width
        elif action == 4:
            x_width = self.boundary["xmax"] - self.boundary["xmin"]
            y_width = self.boundary["ymax"] - self.boundary["ymin"]
            self.boundary["xmin"] = self.boundary["xmax"] - scale * x_width
            self.boundary["ymin"] = self.boundary["ymax"] - scale * y_width
        elif action == 5:
            x_width = self.boundary["xmax"] - self.boundary["xmin"]
            y_width = self.boundary["ymax"] - self.boundary["ymin"]
            self.boundary["xmin"] = self.boundary["xmin"] + (1-scale) * x_width / 2
            self.boundary["ymin"] = self.boundary["ymin"] + (1-scale) * y_width / 2
            self.boundary["xmax"] = self.boundary["xmin"] + scale * x_width
            self.boundary["ymax"] = self.boundary["ymin"] + scale * y_width

        self.sub_image = self.image[int(self.boundary["ymin"]):int(self.boundary["ymax"]),
                         int(self.boundary["xmin"]): int(self.boundary["xmax"])]

    def update_history_vector(self, action):
        # one hot action

        non_zero = len(torch.nonzero(self.history))

        # if not out
        if non_zero < Settings.history_num:
            self.history[non_zero][action-1] = 1.0

        else:
            cur_vector = torch.zeros(Settings.action_num)
            cur_vector[action - 1] = 1.0

            if Settings.cuda:
                cur_vector = cur_vector.cuda()

            for i in range(Settings.history_num - 1):
                self.history[i][:] = self.history[i+1][:]
            self.history[Settings.history_num-1][:] = cur_vector
        return

    def get_state(self):
        image_feature = get_image_feature(self.sub_image)
        history_feature = self.history.view(1, -1)
        state = torch.cat((image_feature, history_feature), 1)
        return state

    def get_next_action(self, q_model, eps):
        """
        get next action based on prediction
        """
        if random.random() < eps:
            action = random.randint(1, 6)
            action = torch.tensor(action)
        else:
            q_values = q_model(self.get_state())
            _, predicted = torch.max(q_values.data, 1)
            action = predicted[0] + 1
        if Settings.cuda:
            action = action.cuda()
        return action




















