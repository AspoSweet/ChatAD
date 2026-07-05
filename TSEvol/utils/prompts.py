import json
actions = ['<raise>', '<answer>']
consultant_instruction = f"""
You are Consultant tasked with generating one time-series-related question based on following seven analytical dimensions:
1. trend: used to describe the variation trends in historical data, including four types: upward, downward, stable, and mixed trends.
2. seasonality: used to describe the periodic fluctuation characteristics of historical data. It includes ten types: non-periodic, sinusoidal, square wave, triangular wave, sawtooth wave, stepwise, composite, random, and segmented patterns.
3. statistics: used to quantify the numerical characteristics of historical data. It includes eight types: mean, variance, standard deviation, maximum, minimum, skewness, unit root test value, and mode.
4. local features: used to identify key points or abnormal behaviors in historical data, including peaks, troughs, mutation points, outliers, spikes, and jumps.
5. multivariate Relationships: the multivariate attribute includes 14 types: covariance matrix, Granger causality, mutual information, Dice coefficient, Pearson correlation coefficient, edit distance, Jaccard similarity, cosine similarity, Euclidean distance, Manhattan distance, DTW (Dynamic Time Warping), Mahalanobis distance, Hamming distance, KL divergence (Kullback-Leibler), Earth Mover’s Distance (EMD), and TAD (Time Series Alignment Distance).
6. compressed representations: compression is used to generate compact representations of raw data by transforming historical data into new forms that enhance complexity. It includes four methods: Discrete Wavelet Transform (DWT), Discrete Fourier Transform (DFT), segmented averaging, and segmented mode extraction.
7. background evolution: Extend and enhance the background context of non-time-series content.
You should generate new questions from the given 'history' instruction data, and each question should be clear, relevant, and designed to guide deeper analysis or understanding of time-series data. Finish your response in English, and question must end with a question mark '?'."""



client_instruction = f"""
You are a time-series expert. Your task is to answer the given question based the 'history' instruction data in a strict, accurate, and concise manner. Finish your response in English. """

intern_instruction = f"""You are a time-series expert. Your task is to answer the given question based on the given 'history' instruction data with deep reasoning if needed, but your response must remain strictly constrained.  """

supervisor_instruction = f""" You need to provide a brief analysis based on the given 'history' instruction data and answers. If the answer is correct, say nothing. If the answer is incorrect, provide a summary of no more than 100 words."""


class JsonTemplateAnswer:
    def __init__(self):
        pass
    def get_final_round_answer_for_client(self):
        return "You must respond ONLY in valid JSON format. Format: {\"answer\": \"'Normal' or 'Anomalous'\", \"summary\": \"A concise and essential analytical process, limited to no more than 150 words.\"}"
    def get_supervisor_answer_in_each_round(self):
        return "You must respond ONLY in valid JSON format. Format: {\"answer\": \"'Normal' or 'False'\", \"advice\": \"If the prediction is correct, return nothing. If incorrect, provide a concise and actionable suggestion, strictly limited to 80 words.\"}"
    def get_client_answer_in_each_round(self):
        return "You must respond ONLY in valid JSON format. Format: {\"answer\": \"Provide an accurate and concise answer to the question, focusing only on the essential information, limited to no more than 200 words.\"}"
    def get_intern_answer_in_each_round(self):
        return "Please determine whether there are anomalies in this time series given information above. You must respond ONLY in valid JSON format. Format: {\"answer\": \"'Normal' or 'Anomalous'\", \"summary\": \"A concise and essential analytical process, limited to no more than 150 words.\"}"
    def get_first_round_question_for_consultant(self):
        return "Please conduct an detailed anomaly detection analysis on the inputted time series data and raise a question with ONLY in valid JSON format. Format: {\"question\": \"A concise and valuable question, strictly limited to 100 words.\", \"summary\": \"Explain why you are asking this question, strictly limited to 200 words.\" }."
    def get_each_round_question_for_consultant(self, A, Q_list):
        data = f"""Please consider the previous Answer and avoid asking duplicate Question List. Question List: {Q_list}, Last Question Reply/Answer {A}"""
        return data + "Please conduct an detailed anomaly detection analysis on the inputted time series data and raise a question with ONLY in valid JSON format. Format: {\"question\": \"A concise and valuable question, strictly limited to 100 words.\", \"summary\": \"Explain why you are asking this question, strictly limited to 200 words.\" }."
    def get_final_round_question_for_consultant(self):
        return "Please determine whether there are anomalies in this time series given information above. You must respond ONLY in valid JSON format. Format: {\"answer\": \"'Normal' or 'Anomalous'\", \"summary\": \"A concise and essential analytical process. \"}"
    def get_final_results_for_client(self, final_flag, source_answer, intern_source_data, debugFlag):
        # 1. parse source_answer
        data = {"answer": "Normal", "summary": "None"}
        upper_source_answer = source_answer.upper().split()
        if 'ANOMALY' in upper_source_answer or 'ANOMALOUS' in upper_source_answer:
            data["answer"] = "Anomalous"
        # 2. parse the feedback from the Intern
        if final_flag == False:
            data["summary"] = f"""\"{source_answer}\""""
        if final_flag == True and intern_source_data is not None:   # the returned intern_source_data is not empty
            answers = intern_source_data["answer"]
            answers =  json.loads(answers)
            cotList = intern_source_data["cotList"]
            cot = "\n".join(cotList)
            data["summary"] = f"""\"{answers["summary"]}\n<think>{cot}</think>\""""
            #data["answer"] = f"""\"{answers["answer"]}\""""
        # 3. assemble the message
        return data




class Prompter:
    def __init__(self):
        self.consultant_instruction = consultant_instruction
        self.client_instruction = client_instruction
        self.intern_instruction = intern_instruction
        self.supervisor_instruction = supervisor_instruction
