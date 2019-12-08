from Image import Image
from Agent import Agent
from IoU import *
from Models import QModel
import torch


class ObjectTest:
    def __int__(self, data, model_name):
        """
        take in a FileUtils
        """
        self.data = data
        self.model = QModel()
        self.model.load_state_dict(torch.load("{}{}".format(Settings.model_path, model_name)))
        if Settings.cuda:
            self.model.cuda()
        self.model.eval()
        self.device = torch.device("cuda" if Settings.cuda else "cpu")

    def localization(self, index):
        image = Image(self.data, index)
        annotation_list = image.objects
        agent = Agent(image.image)
        done = False
        old_iou_list = None
        reward = 0

        for step in range(Settings.max_step):
            print("Step {}".format(step), end="=>")

            iou_list = [iou_calculator(agent.boundary, x) for x in annotation_list]
            max_index = max(range(len(iou_list)), key=lambda x: iou_list[x])
            iou = max(iou_list)

            print("current iou = {}".format(iou), end=" || ")
            old_iou = old_iou_list[max_index] if old_iou_list else 0
            old_iou_list = iou_list

            # determine if we should end the result
            if iou > Settings.iou_threshold:
                action = torch.tensor(6).to(self.device)
            else:
                action = agent.get_next_action(self.model, eps=0)

            if action == 6:
                reward = reward_terminal(iou)
                done = True
            else:
                agent.hierarchical_move(action)
                agent.update_history_vector(action)
                image.draw_one_box(agent.boundary)
                image.add_text(step, (agent.boundary["xmin"], agent.boundary["ymin"]))
                if agent.sub_image.shape[0] * agent.sub_image.shape[1] == 0:
                    done = True
                else:
                    reward = reward_move(old_iou, iou)

            print("current action = {}".format(int(action)), end=" || ")
            print("current reward = {}".format(float(reward)))
            if done:
                break
        image.show()
