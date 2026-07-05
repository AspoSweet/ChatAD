"""
Describe: logging utilities for the TSEvol pipeline
"""
import time, os, sys, csv, json, ast
import pandas as pd
from triton.language.semantic import truediv


class FileManager:
    def __init__(self, saved_path):
        self.saved_path = saved_path
        self.timestamp = self.create_father_dir()
        self.father_dir = os.path.join(saved_path, self.timestamp)
        self.file_initializer_flag = False
        self.raw_data_file = os.path.join(self.father_dir, "raw_data.csv")                      # raw inter-agent conversation log (CSV)
        self.conversation_data_file = os.path.join(self.father_dir, "conversation_data.csv")    # curated multi-turn conversation log (CSV)



    def create_father_dir(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.saved_path, timestamp)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f"Wrong For Father Dir: {e}")
            return None
        return timestamp

    def data_writer(self, new_data, file_type):
        with open(file_type, mode="a", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(new_data)



class ToolsPackage:
    def __init__(self):
        pass

    def read_json_from_source_ad(self, file_path):
        domains= []
        answers = []
        questions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data:
                domains.append(d['domain'])
                questions.append(d['question'])
                answers.append(d['answer'])
            return {'domains':domains, 'answers':answers, 'questions':questions}

    def get_last_index(self, last_log_file_path = None):
        if last_log_file_path is None:
            return 0
        else:
            data = pd.read_csv(last_log_file_path, header=None)
            data = pd.DataFrame(data)
            last_row_dict = ast.literal_eval(data.iloc[-1, 0])
            last_id = last_row_dict.get('id')
            print("The Last id is:", last_id)
            return last_id + 1
        return 0

    def parse_each_round_question_for_consultant(self, datas, debugFlag):
        data = datas["answer"]     # extract the "answer" field from the JSON payload
        if not data or not isinstance(data, str):
            print("Wrong For Parse First Round Question For Consultant!")
            exit(0)
        try:
            data = json.loads(data)
            if debugFlag:
                print()
                print("\nThe sub-question from consultant is: ", data['question'])
            return data["question"]
        except json.JSONDecodeError as e:
            print("Wrong Parse For Consultant: ", e)
            exit(0)

    def parse_first_round_question_for_consultant(self, datas, debugFlag):
        data = datas["answer"]     # extract the "answer" field from the JSON payload
        if not data or not isinstance(data, str):
            print("Wrong For Parse First Round Question For Consultant!")
            exit(0)
        try:
            data = json.loads(data)
            if debugFlag:
                print()
                print("\nThe first question from consultant is: ", data['question'])
            return data["question"]
        except json.JSONDecodeError as e:
            print(data)
            print("Wrong Parse: ", e)
            exit(0)

    def parse_client_answer_in_each_round(self, datas, debugFlag):
        data = datas["answer"]
        cot = datas["cotList"]
        if not data or not isinstance(data, str):
            print("Wrong For Parse Client Answer in Each-Round!")
            exit(0)
        try:
            data = json.loads(data)
            if len(cot) == 0:
                cot = "None"
            else:
                cot = "\n".join(cot)
            if debugFlag:
                print()
                print("\nThe generate answer from client: ", data['answer'])
                #print(f"""###Cot: \n{cot}""")
            return data['answer'], cot
        except json.JSONDecodeError as e:
            print("Wrong Parse: ", e)
            exit(0)

    def parse_intern_answer_in_each_round(self, datas, debugFlag):
        data = datas["answer"]
        if not data or not isinstance(data, str):
            print("Wrong For Parse Client Answer in Each-Round!")
            exit(0)
        try:
            data = json.loads(data)
            if debugFlag:
                print()
                print("\nThe verify from client: ", data['answer'])

            return data['answer'], data["summary"]
        except json.JSONDecodeError as e:
            print("Wrong Parse: ", e)
            exit(0)

    def parse_answer_for_supervisor_in_each_round(self, ground_truth, predicted_answers, debugFlag):
        data = predicted_answers["answer"]
        answer = False
        if not data or not isinstance(data, str):
            print("Wrong For Parse Client Answer in Each-Round!")
            exit(0)
        try:
            data = json.loads(data)["answer"]
            predicted = data.upper().split()
            ground_truth = ground_truth.upper().split()
            truth_list = ["ANOMALY", "ANOMALOUS"]
            faulth_list = ["NORMAL"]
            if ('ANOMALY' in predicted or 'ANOMALOUS' in predicted) and ('ANOMALOUS' in ground_truth or 'ANOMALY' in ground_truth):
                answer = True
            elif "NORMAL" in predicted and "NORMAL" in ground_truth:
                answer = True
            else:
                answer = False

            if debugFlag:
                print()
                print(f"""\nThe verify from supervisor:  \npredicted from intern is : {predicted}, \nground_truth is: {ground_truth}, \nthe supervisor's feedback is:{answer}""")


        except json.JSONDecodeError as e:
            print("Wrong Parse: ", e)
            exit(0)
        return answer






