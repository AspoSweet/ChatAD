import argparse
from utils.person import Consultant, Client, Intern, Supervisor
from utils.prompts import Prompter, JsonTemplateAnswer
from utils.api_utils import Model
from utils.tools import ToolsPackage, FileManager
import jsonlines
import json, time, re


class TSEvol(ToolsPackage, FileManager, JsonTemplateAnswer):
    def __init__(self, data_path, prompter, consultant, client, intern, supervisor, debug = False, human_feedback =  False, rounds_number = 3, saved_path = './saved_path',last_log_file=None):
        super().__init__()
        FileManager.__init__(self, saved_path=saved_path)
        self.debug = debug
        self.human_feedback = human_feedback
        self.rounds_number = rounds_number
        self.history_conversation_data = None
        self.history_raw_data = None
        self.consultant = consultant
        self.client = client
        self.intern = intern
        self.supervisor = supervisor
        self.prompter = prompter
        self.data_path = data_path
        self.last_log_file = last_log_file
        self.total_tokens = [0, 0]        # [input_tokens, output_tokens]
        self.client_tokens = [0, 0]
        self.intern_tokens = [0, 0]
        self.consultant_tokens = [0, 0]
        self.supervisor_tokens = [0, 0]
        self.saved_path = saved_path
    def history_initilize(self):
        self.history_conversation_data = {  # stores the pre-processed conversation records
            'id': -1,
            'rounds': -1,
            'tokens': -1,
            'conversations': [],
        }
        self.history_raw_data = {  # stores the raw inter-agent conversation records
            'id': -1,
            'rounds': -1,
            'tokens': -1,
            'conversations': [],
        }
        self.client_tokens = [0, 0]
        self.intern_tokens = [0, 0]
        self.consultant_tokens = [0, 0]
        self.supervisor_tokens = [0, 0]
    def token_summary(self, person, returned_data, total_tokens):
        if person == 'Consultant':
            self.consultant_tokens[0], self.consultant_tokens[1] = returned_data["token_used"][0] + self.consultant_tokens[0], returned_data["token_used"][1] +  self.consultant_tokens[1]
            total_tokens[0], total_tokens[1] = returned_data["token_used"][0] + total_tokens[0], returned_data["token_used"][1] + total_tokens[1]
        elif person == 'Client':
            self.client_tokens[0], self.client_tokens[1] = returned_data["token_used"][0] + self.client_tokens[0], returned_data["token_used"][1] + self.client_tokens[1]
            total_tokens[0], total_tokens[1] = returned_data["token_used"][0] + total_tokens[0], returned_data["token_used"][1] + total_tokens[1]
        elif person == 'Intern':
            self.intern_tokens[0], self.intern_tokens[1] = returned_data["token_used"][0] + self.intern_tokens[0], returned_data["token_used"][1] + self.intern_tokens[1]
            total_tokens[0], total_tokens[1] = returned_data["token_used"][0] + total_tokens[0], returned_data["token_used"][1] + total_tokens[1]
        else:
            pass
        return total_tokens

    def add_histroy(self, person, content1, content2, write_flag=0):
        if write_flag == 0:  # 0: write to the raw log only; the current result will be curated later
            self.history_raw_data["conversations"].append({"role": str(person), "content": content1})
        else:
            self.history_conversation_data["conversations"].append( {"role": str(person), "content": content2})  # append the new question to the conversation record


    def tsevol_instruction(self):
        data = self.read_json_from_source_ad(self.data_path) #{domain, answer, input}
        begin_index = self.get_last_index(self.last_log_file)
        end_index = len(data['answers'])
        print(f"""[Begin, End] : [{begin_index}, {end_index}]""")

        for i in range(begin_index, end_index):
            rounds = self.rounds_number
            question_list = []
            self.history_initilize()
            total_tokens = [0, 0]
            self.history_conversation_data["id"] = i
            self.history_raw_data["id"] = i
            time_begin = time.time()
            if self.debug:
                print("*"*100)
                print(f"""\n1 Begin the {i}-th sample's evolution.""")
                print(f"""\n  Rounds: {rounds}""")
            source_question = data['questions'][i]
            source_answer = data['answers'][i]

            # Step 1: the Consultant generates the first-round question Q_0
            if rounds - 1 > 0:     # reserve the final round for the closing QA pair
                if self.debug:
                    print(f"""\n2 Begin Consultant Generate the 0-th Question.""")
                    print(f"""\n  Rounds: {rounds}""")
                self.consultant.chat_history.append({"role" : "user", "content" : source_question + "\n" + self.get_first_round_question_for_consultant()})
                returned_data_consultant = self.consultant.Model.get_parsed_response_from_model(f"""{self.consultant.chat_history[0]["content"]}\n{self.consultant.chat_history[-1]["content"]}""")
                new_question_Q0 = self.parse_first_round_question_for_consultant(returned_data_consultant, self.debug)
                self.add_histroy(self.consultant.character, f"""{source_question}\n{new_question_Q0}""", None, write_flag=0)
                total_tokens = self.token_summary(self.consultant.character, returned_data_consultant, total_tokens)
                self.consultant.chat_history.pop()


            # Step 2: the Client answers in a loop over rounds to ensure correctness
            if self.debug:
                print(f"""\n3 Begin Client, Intern, Supervisor verify the 0-th Question.""")
                print(f"""\n  Rounds: {rounds}""")
            false_flag, vaild_round, final_flag, final_pesponse = 0, 0, False, None
            while rounds - 1 > 0:    # reserve the final round for the closing QA pair
                rounds -= 1
                question_list.append(new_question_Q0)
                # Step 2.1: the Client produces an answer
                self.client.chat_history.append({"role": "user", "content": f""" {source_question}\n{new_question_Q0}\n{self.get_client_answer_in_each_round()}"""})
                returned_data_client = self.client.Model.get_parsed_response_from_model(self.client.chat_history[0]["content"] + "\n" + self.client.chat_history[-1]["content"])
                new_answer_client, new_cot_client = self.parse_client_answer_in_each_round(returned_data_client, self.debug)
                total_tokens = self.token_summary(self.client.character, returned_data_client, total_tokens)
                self.add_histroy(self.client.character, new_answer_client, None, write_flag=0)
                self.client.chat_history.pop()

                # Step 2.2: the Intern assesses the answer
                self.intern.chat_history.append({"role": "user", "content": f"""{source_question}\n{self.get_intern_answer_in_each_round()}\n(some background: {new_answer_client})"""})
                returned_data_intern = self.intern.Model.get_parsed_response_from_model(self.intern.chat_history[0]["content"] + "\n" + self.intern.chat_history[-1]["content"])
                self.add_histroy(self.intern.character, returned_data_intern, None, write_flag=0)
                total_tokens = self.token_summary(self.intern.character, returned_data_intern, total_tokens)
                self.intern.chat_history.pop()

                # Step 2.3: the Supervisor verifies the answer
                answer_flag = self.parse_answer_for_supervisor_in_each_round(source_answer, returned_data_intern, self.debug)
                self.add_histroy(self.supervisor.character, answer_flag, None, write_flag=0)

                # Step 2.4: decide from the Supervisor's feedback whether to keep this result
                if answer_flag == True:
                    if self.debug:
                        print(f"""\n The generated Question Q_0 very good.""")
                        print(f"""\n  Rounds: {rounds}""")
                    vaild_round += 1                                                                                     # valid evolution result +1
                    final_pesponse = returned_data_intern
                    self.add_histroy(self.consultant.character, None, f""""{source_question}\n{new_question_Q0}""", write_flag=1)
                    self.add_histroy(self.client.character, None, f"""{new_answer_client}\n<think>{new_cot_client}</think>""", write_flag=1)
                    break
                else:
                    if self.debug:
                        print(f"""\n The generated Question Q_0 very bad. Re generating...""")
                        print(f"""\n  Rounds: {rounds}""")
                    false_flag += 1  # invalid evolution result +1; regenerate question Q_0
                if rounds >= 0:
                    self.consultant.chat_history.append({"role": "user","content": source_question + "\n" + self.get_first_round_question_for_consultant()})
                    returned_data_consultant = self.consultant.Model.get_parsed_response_from_model(f"""{self.consultant.chat_history[0]["content"]}\n{self.consultant.chat_history[-1]["content"]}""")
                    new_question_Q0 = self.parse_first_round_question_for_consultant(returned_data_consultant,self.debug)
                    self.add_histroy(self.consultant.character, new_question_Q0, None, write_flag=0)
                    total_tokens = self.token_summary(self.consultant.character, returned_data_consultant, total_tokens)
                    self.consultant.chat_history.pop()

            # Step 3: generate the intermediate-round QA pairs
            # Step 3.1: the Consultant generates question Q_k
            if rounds - 1 > 0:  # reserve the final round for the closing QA pair
                if self.debug:
                    print(f"""\n 4 Begin generate the sub Questions.""")
                    print(f"""\n  Rounds: {rounds}""")
                self.consultant.chat_history.append({"role": "user", "content": f"""{source_question}\n{self.get_each_round_question_for_consultant(new_answer_client,question_list)}"""})
                returned_data_consultant = self.consultant.Model.get_parsed_response_from_model(f"""{self.consultant.chat_history[0]["content"]}\n{self.consultant.chat_history[-1]["content"]}""")
                new_question_Qk = self.parse_each_round_question_for_consultant(returned_data_consultant, self.debug)
                self.add_histroy(self.consultant.character, new_question_Qk, None, write_flag=0)
                total_tokens = self.token_summary(self.consultant.character, returned_data_consultant, total_tokens)
                self.consultant.chat_history.pop()

            # Step 3.2: generate and verify in a loop
            while rounds - 1 > 0: # reserve the final round for the closing QA pair
                rounds -= 1
                question_list.append(new_question_Qk)
                # Step 3.2.1: the Client answers the new question
                if self.debug:
                    print(f"""\n Begin the {vaild_round} Q-A Pair.""")
                    print(f"""\n Rounds: {rounds}""")
                self.client.chat_history.append({"role": "user", "content": f""" {source_question}\n{new_question_Qk}\n{self.get_client_answer_in_each_round()}"""})
                returned_data_client = self.client.Model.get_parsed_response_from_model(self.client.chat_history[0]["content"] + "\n" + self.client.chat_history[-1]["content"])
                new_answer_client, new_cot_client = self.parse_client_answer_in_each_round(returned_data_client, self.debug)
                total_tokens = self.token_summary(self.client.character, returned_data_client, total_tokens)
                self.client.chat_history.pop()
                self.add_histroy(self.client.character, new_answer_client, None, write_flag=0)

                # Step 3.2.2: the Intern assesses the answer
                self.intern.chat_history.append({"role": "user","content": f"""{source_question}\n{self.get_intern_answer_in_each_round()}\n(some background: {new_answer_client})"""})
                returned_data_intern = self.intern.Model.get_parsed_response_from_model(self.intern.chat_history[0]["content"] + "\n" + self.intern.chat_history[-1]["content"])
                self.add_histroy(self.intern.character, returned_data_intern, None, write_flag=0)
                total_tokens = self.token_summary(self.intern.character, returned_data_intern, total_tokens)
                self.intern.chat_history.pop()

                # Step 3.2.3: the Supervisor verifies the answer
                answer_flag = self.parse_answer_for_supervisor_in_each_round(source_answer, returned_data_intern, self.debug)
                self.add_histroy(self.supervisor.character, answer_flag, None, write_flag=0)

                # Step 2.4: decide from the Supervisor's feedback whether to keep this result
                if answer_flag == True:
                    vaild_round += 1  # valid evolution result +1
                    final_pesponse = returned_data_intern
                    self.add_histroy(self.consultant.character, None, f"""{new_question_Qk}""", write_flag=1)
                    self.add_histroy(self.client.character, None,f"""{new_answer_client}\n<think>{new_cot_client}</think>""", write_flag=1)
                else:
                    false_flag += 1  # invalid evolution result +1; regenerate question Q_0
                if rounds >= 0:
                    self.consultant.chat_history.append({"role": "user", "content": f"""{source_question}\n{self.get_each_round_question_for_consultant(new_answer_client, question_list)}"""})
                    returned_data_consultant = self.consultant.Model.get_parsed_response_from_model( f"""{self.consultant.chat_history[0]["content"]}\n{self.consultant.chat_history[-1]["content"]}""")
                    new_question_Qk = self.parse_first_round_question_for_consultant(returned_data_consultant,self.debug)
                    self.add_histroy(self.consultant.character, new_question_Qk, None, write_flag=0)
                    total_tokens = self.token_summary(self.consultant.character, returned_data_consultant, total_tokens)
                    self.consultant.chat_history.pop()

            # Step 4: produce the final-round QA pair
            if len(self.history_conversation_data["conversations"]) >=2:
                final_flag = True


            if final_flag == False:
                self.add_histroy(self.consultant.character, f"""{self.get_final_round_question_for_consultant()}""", None, write_flag=0)
                self.add_histroy(self.consultant.character, None, f"""{source_question}\n{self.get_final_round_question_for_consultant()}""", write_flag=1)
            else:
                self.add_histroy(self.consultant.character, self.get_final_round_question_for_consultant(), None, write_flag=0)
                self.add_histroy(self.consultant.character, None, self.get_final_round_question_for_consultant(), write_flag=1)
            fina_answer_for_clinet = self.get_final_results_for_client(final_flag, source_answer, final_pesponse,self.debug)
            self.add_histroy(self.client.character, fina_answer_for_clinet, None, write_flag=0)
            self.add_histroy(self.client.character, None, fina_answer_for_clinet, write_flag=1)

            # print the summary of the current round
            time_end = time.time()
            self.history_raw_data["rounds"], self.history_conversation_data["rounds"] =  vaild_round + 1, vaild_round + 1
            self.history_raw_data["tokens"], self.history_conversation_data["tokens"] = total_tokens, total_tokens
            self.data_writer([self.history_conversation_data], self.conversation_data_file)
            self.data_writer([self.history_raw_data], self.raw_data_file)
            print(f"""The Time Useage: {time_end - time_begin} Seconds.""")
            print(f"""The Total Tokens Used: {total_tokens}.""")
            print(json.dumps(self.history_conversation_data, indent=4, ensure_ascii=False))
            #print(json.dumps(self.history_raw_data, indent=4, ensure_ascii=False))
            




def run_tsevol(data_path, saved_path, rounds, last_log_path=None):
    # The backend LLM is configured via environment variables;
    # see TSEvol/utils/api_utils.py for the full list of options.
    promptor = Prompter()    # build the prompter
    consultant_model = Model()
    consultant = Consultant(promptor, consultant_model)
    client_model = Model()
    client = Client(promptor, client_model)
    intern_model = Model()
    intern = Intern(promptor, intern_model)
    supervisor_model = Model()
    supervisor = Supervisor(promptor, supervisor_model)
    tsevol = TSEvol(data_path, promptor, consultant, client,intern, supervisor,debug=True, human_feedback=False, rounds_number=rounds, last_log_file=last_log_path, saved_path=saved_path)
    tsevol.tsevol_instruction()


if __name__ == "__main__":
    N = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./TSEvol_AD_Source_Part_11.json")
    parser.add_argument("--saved_path", type=str, default="./saved_un_pathed")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--last_log_path", type=str, default=None)
    args = parser.parse_args()
    print('Arguments: ', args)
    run_tsevol(args.data_path, args.saved_path, args.rounds, args.last_log_path)

