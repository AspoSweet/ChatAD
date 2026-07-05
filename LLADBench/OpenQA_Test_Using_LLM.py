"""LLADBench: scoring open-ended QA answers with an LLM judge."""
from openai import AzureOpenAI, OpenAI
import time, json, openai, os
from openai import RateLimitError
from openai import OpenAIError
import pandas as pd
import argparse

# The judge LLM is configured via environment variables (see README):
#   Azure OpenAI: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY /
#                 AZURE_OPENAI_API_VERSION / LLM_DEPLOYMENT
#   OpenAI:       LLM_PROVIDER=openai, OPENAI_API_KEY, optional OPENAI_BASE_URL
api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-03-01-preview')
deployment_name = os.getenv('LLM_DEPLOYMENT', 'gpt-5')


def _build_client():
    provider = os.getenv('LLM_PROVIDER', 'azure').lower()
    if provider == 'azure':
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        if not endpoint or not api_key:
            raise EnvironmentError('Please set AZURE_OPENAI_ENDPOINT and '
                                   'AZURE_OPENAI_API_KEY (or LLM_PROVIDER=openai '
                                   'with OPENAI_API_KEY).')
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                           api_version=api_version)
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                  base_url=os.getenv('OPENAI_BASE_URL') or None)

API_MAX_RETRY = 3
API_ERROR_OUTPUT = "$ERROR$"
ANSWER_ERROR_OUTPUT = "$ANSWER_ERROR$"

SYSTEM_PROMPT = """
You are a time-series analysis expert. Your role is to rigorously and strictly evaluate a student's answer against the given question and the ground truth.

Scoring Rules:

- Fully correct → return 1
- Completely incorrect → return 0
- Partially correct → return a decimal between 0 and 1 (e.g., 0.75)

Output Requirements:

- Output only one numeric value in the format: <score>
- Value must be in the range [0, 1], decimals allowed
- Do not include explanations or any additional fields

Inputs:

- Original question: Q
- Student answer: P
- Ground truth: Ground truth

Evaluation Criteria (more lenient):

Focus on core time-series concepts: trend, seasonality, anomalies, volatility, temporal patterns, units, directionality
- If the answer is generally correct but lacks details → 0.6–0.8
- If the answer has some errors but reasoning is reasonable → 0.3–0.5
- If the answer is mostly off-topic but mentions relevant terms → 0.1–0.2
- Do not penalize too strictly; allow some ambiguity
"""

class Model:
    def __init__(self, deployment_name, instance, endpoint, api_version):
        """
        :param deployment_name: deployment/model name of the judge LLM
        """
        self.deployment_name = deployment_name
        self.instance = instance
        self.endpoint = endpoint
        self.api_version = api_version
        self.client = _build_client()
        

    def parse_response_GPT5(self, response):    # parse a Responses-API result
        cot_list = [data.text for data in response.output[0].summary]
        answer = response.output[1].content[0].text
        tokens_used = [response.usage.input_tokens, response.usage.output_tokens]
        return cot_list, answer, tokens_used
    def get_parsed_response_from_model(self, content):
        response = API_ERROR_OUTPUT
        cotList, answer, token_used = ANSWER_ERROR_OUTPUT, ANSWER_ERROR_OUTPUT, 0
        if self.deployment_name in ["gpt-5_2025-08-07"]:  # request format differs across model families
            for retry_i in range(API_MAX_RETRY):
                try:
                    response = self.client.responses.create(
                        model=self.deployment_name,
                        store=True,
                        reasoning={"effort": "medium", "summary": "auto"},
                        text={"verbosity": "medium"},
                        input=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                            "role": "user",
                            "content": content}],
                    )
                    cotList, answer, token_used = self.parse_response_GPT5(response)  # returns the CoT list and the final answer
                    return cotList, answer, token_used

                except openai.OpenAIError as e:  # retry up to API_MAX_RETRY times
                    #print(f"""{Model.deployment_name} encountered error""")
                    print(type(e), e)
                    exit(0)
                time.sleep(60)

    

class OpenQATestUsingLLM:
    def __init__(self, input_path, ground_truth_path, save_path):
        self.ground_truth_file = ground_truth_path
        self.save_path = save_path
        self.input_path = input_path
        self.log_path = save_path + ".log" 
        self.GPT5 = Model(deployment_name, instance, endpoint, api_version)
        self.score_list = []
        self.input_token = 0
        self.output_token = 0
    def parse_file_from_path(self):
        questions, answers, predicted = [], [], []
        with open(self.input_path, 'r', encoding='utf-8') as f:   
            for line in f:
                line = line.strip()  
                if line:  
                    obj = json.loads(line)  
                    questions.append(obj['prompt'])
                    answers.append(obj['label'])
                    predicted.append(obj['predict'])
        return questions, answers, predicted

    def parse_ground_truth(self):
        ground_truths = []
        questions = []

        with open(self.ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)  # data is a list[dict]

        for i in range(len(data)):
            ground_truths.append(data[i]['conversations'][-1]['value'])
            questions.append(data[i]['conversations'][0]['value'])
        return questions, ground_truths
    
    def parse_begin_point_from_log(self):
        begin_point = 0
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        evaluation_score = float(obj['evaluation_score'])
                        self.score_list.append(evaluation_score)
                        self.input_token += obj.get('tokens_used', [0,0])[0]
                        self.output_token += obj.get('tokens_used', [0,0])[1]
                        print(f"Loaded log for sample index {obj['index']} with score {evaluation_score}")
                        begin_point += 1
        return begin_point
    
    def run_openQA_on_data(self):
        questions, answers = self.parse_ground_truth()
        begin_point = self.parse_begin_point_from_log()
        questions_, answers_, predicted = self.parse_file_from_path()
        number_of_samples = len(questions)
        for i in range(begin_point, number_of_samples):
            print("-" * 50)
            print(f"Processing sample {i+1}/{number_of_samples}")
            Q = questions[i]
            A = answers[i]
            P = predicted[i]
            content = f"""Original question (Q): {Q}; Student answer (P): {P}; Ground truth (A): {A}"""
            cotList, answer, token_used = self.GPT5.get_parsed_response_from_model(content)
            datadict = {
                "index": i,
                "question": Q,
                "ground_truth": A,
                "predicted_answer": P,
                "cotList": cotList,
                "evaluation_score": answer,
                "tokens_used": token_used
            }
            self.score_list.append(float(answer))
            self.input_token += token_used[0]
            self.output_token += token_used[1]
            datadict_json = json.dumps(
                    datadict,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,            # keep insertion order
                    separators=(",", ": ")      # compact separators
                )
            
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(datadict, ensure_ascii=False) + "\n")
            print(datadict_json)
        # saving anwsers
        if len(self.score_list) >= number_of_samples:   
            average_score = sum(self.score_list) / len(self.score_list)
            saved_data = {
                "average_evaluation_score": average_score,
                "total_samples": len(self.score_list),
                "total_input_tokens": self.input_token,
                "total_output_tokens": self.output_token,
                "input_token_cost": self.input_token * 1.25 / 1000000,  
                "output_token_cost": self.output_token  * 10 / 1000000
            }
            print(json.dumps(
                    saved_data,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,           
                    separators=(",", ": ")      
                ))
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(saved_data, f, ensure_ascii=False, indent=2)
        else:
            print("No evaluation scores were recorded.")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_file", type=str, default="~generated_predictions.jsonl")
    parser.add_argument("--output_file", type=str, default="~LLADBench-Reasoning-Results1.json")
    parser.add_argument("--ground_truth_file", type=str, default="~OpenQA-Test-1K.json")
    args = parser.parse_args()
    Test = OpenQATestUsingLLM(input_path=args.predicted_file, ground_truth_path=args.ground_truth_file, save_path=args.output_file)
    Test.run_openQA_on_data()