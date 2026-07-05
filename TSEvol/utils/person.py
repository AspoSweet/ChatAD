"""
Describe: A python source file for persons
"""

from utils.api_utils import get_parsed_response
from utils.api_utils import Model
import re


class Person:
    def __init__(self, Promptor, Model):
        self.Model = Model
        self.character = "NONE"                        # role name of this agent
        self.model_name = Model.deployment_name        # name of the backend LLM
        self.promptor = Promptor                      # initial prompts for the four agent roles
        self.chat_history = []                         # chat history

    def add_history(self, role, content, action_guide=''):
        self.chat_history.append({"role": role, "content": content})


class Consultant(Person):
    def __init__(self, Promptor, Model):
        super().__init__(Promptor, Model)
        self.model_name = Model.deployment_name
        self.character = "Consultant"
        self.chat_history = [{"role": "system", "content": self.promptor.consultant_instruction}]    #


class Client(Person):
    def __init__(self, Promptor, Model):
        super().__init__(Promptor, Model)
        self.model_name = Model.deployment_name
        self.character = "Client"
        self.chat_history = [{"role": "system", "content": self.promptor.client_instruction}]      # sets the initial system persona


class Intern(Person):
    def __init__(self, Promptor, Model):
        super().__init__(Promptor, Model)
        self.model_name = Model.deployment_name
        self.character = "Intern"
        self.chat_history = [{"role": "system", "content": self.promptor.intern_instruction}]

class Supervisor(Person):
    def __init__(self, Promptor, Model):
        super().__init__(Promptor, Model)
        self.model_name = Model.deployment_name
        self.character = "Supervisor"
        self.chat_history = [{"role": "system", "content": self.promptor.supervisor_instruction}]
    def get_ad_task_results(self, ground_truth, predicted_value):
        """
        :param ground_truth:     ground-truth anomaly label
        :param predicted_value:  predicted label
        :return:
        """
        if not isinstance(predicted_value, str) or not isinstance(ground_truth, str):
            raise ValueError("Inputs must be strings.")

        anomaly_labels = {"ANOMALY", "ANOMALOUS"}
        normal_labels = {"NORMAL", "FALSE"}

        if predicted_value in anomaly_labels and ground_truth in anomaly_labels:
            return True
        elif predicted_value in normal_labels and ground_truth in normal_labels:
            return True
        else:
            print(f"Unexpected label combination: predicted={predicted_value}, ground_truth={ground_truth}")
            return False


