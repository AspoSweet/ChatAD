"""
Describe: This script evaluates the performance of anomaly detection models on a given dataset.
"""
import requests
import argparse, json, os, re, ast, time
from collections import defaultdict


class Tools:
    def __init__(self, source_data_apath):
        self.source_data_apath = source_data_apath
        

    def read_data_from_llamafactory_generated_predictions(self):
        datas = []
        with open(self.source_data_apath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                datas.append(data)
        return datas
    
class AD_Task_Evaluation(Tools):    # evaluation on the anomaly detection benchmark
    def __init__(self, source_data_apath, saved_data_path):
        Tools.__init__(self, source_data_apath)
        self.saved_data_path = saved_data_path
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.positive_sumpales = 0
        self.negative_samples = 0
        self.total_samples = 0
        self.wrong_list = []
        self.wrong_list_reason = []
        self.samples = 0
    
    def compute_metrics_two_class(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Accuracy": round(accuracy, 5),
            "Precision": round(precision, 5),
            "Recall": round(recall, 5),
            "FPR": round(fpr, 5),
            "F1-score": round(f1, 5),
            "Samples": self.samples,
            "Anomaly number": self.positive_sumpales,
            "Normal number": self.negative_samples,
            "TP number": self.tp,
            "TN number": self.tn,
            "FP number": self.fp,
            "FN number": self.fn,
            "Effient Sanples": self.tp + self.tn + self.fp + self.fn,
            "Wrong number": len(self.wrong_list),
            "Wrong list": self.wrong_list,
            "Wrong reason": self.wrong_list_reason,
            "Result Confidence": round((self.tp + self.tn + self.fn + self.fp) / self.samples, 5)
        }

    def get_statistics_two_class(self, dealed_label, dealed_prediction):
        flag = True
        if dealed_label == "ANOMALY":
            self.positive_sumpales += 1
            if dealed_prediction == "ANOMALY":
                self.tp += 1
                print("TP detected.")
            elif dealed_prediction == "NORMAL":
                self.fn += 1
                print("FN detected.")
            else:
                print("Invalid prediction:", dealed_prediction)
                flag = False
        elif dealed_label == "NORMAL":
            self.negative_samples += 1
            if dealed_prediction == "ANOMALY":
                self.fp += 1
                print("FP detected.")
            elif dealed_prediction == "NORMAL":
                self.tn += 1
                print("TN detected.")
            else:
                print("Invalid prediction:", dealed_prediction)
                flag = False
        else:
            print("Invalid label:", dealed_label)
            flag = False
        self.total_samples += 1
        return flag
        
    def generate_two_calssification_results(self, label, prediction):
        if label is None:
            label = "INVALID"
        if prediction is None:
            prediction = "INVALID"
        label = label.strip().upper()[0:50]
        #print(label)
        #time.sleep(1)
        prediction = prediction.strip().upper()
        anomaly_answers = {"ANOMALY", "ANOMALOUS"}
        normal_answers = {"NORMAL"}
        if any(ans in label for ans in anomaly_answers) and not any(ans in label for ans in normal_answers):
            dealed_label = "ANOMALY"
        elif any(ans in label for ans in normal_answers) and not any(ans in label for ans in anomaly_answers):
            dealed_label = "NORMAL"
        else:
            print("Invalid label:", label)
            dealed_label = "INVALID"

        if any(ans in prediction for ans in anomaly_answers) and not any(ans in prediction for ans in normal_answers):
            dealed_prediction = "ANOMALY"
        elif any(ans in prediction for ans in normal_answers) and not any(ans in prediction for ans in anomaly_answers):
            dealed_prediction = "NORMAL"
        else:
            print("Invalid prediction:", prediction)
            dealed_prediction = "INVALID"
            
        print("Label:", label)
        print("Dealed Label:", dealed_label)
        print("Prediction:", prediction)
        print("Dealed Prediction:", dealed_prediction)

        flag = self.get_statistics_two_class(dealed_label, dealed_prediction)
        return flag
        

        
    def parse_single_generated_predictions(self, sample, id):
        print(id, "*"*50)
        Label = sample['label']
        #Label = self.Qwen25_deal(Label)
        #Label = ast.literal_eval(Label)
        #Label = Label.get('answer', None)
        Prediction = sample['predict']
        #print("Raw Prediction : ", Prediction)
        #print()
        #print("Sample:", sample)
        try:
            Prediction_JSON = json.loads(Prediction)
            print("Successfully parsed JSON:")
            if "answer" not in Prediction_JSON:
                raise KeyError("Missing 'answer' key in prediction")    
            Answer = Prediction_JSON["answer"]
            
            
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed, trying string method...")
            try:
                match = re.search(r"'answer'\s*:\s*'([^']+)'", Prediction)
                #print(match)
                if match:
                    Answer = match.group(1)
                    print("Successfully extracted answer using regex.")
                    #print("Answer****:", Answer)
                    #print("Label***:", Label)
                    #time.sleep(10)
                else:
                    print("Regex extraction failed.")
                    raise ValueError("Could not extract 'answer' from prediction string")
            except Exception as e2:
                print(f"String method also failed: {str(e2)}")
                #print(Prediction)
                # continue with fallback parsing:
                Prediction = Prediction.upper()
                #print(Prediction)
                if "ANOMALY" in Prediction or "ANOMALOUS" in Prediction:
                    print("Find ANOMALY")
                    Answer =  "ANOMALY"
                elif "NORMAL" in Prediction:
                    print("FIND NORMAL")
                    Answer = "NORMAL"
                else:
                    Answer = None
                
                    
                Label, Answer = Label, Answer
                
        except KeyError as e:
            print(f"Key error: {str(e)}")
            exit(0)
            Label, Answer = None, None
        
        
        flag = self.generate_two_calssification_results(Label, Answer)
        if not flag:   # parsing failed
            self.wrong_list.append(id)
            self.wrong_list_reason.append(Prediction)
            print("Added to wrong list.")
            print(Prediction)
    
    def run(self):
        raw_data= self.read_data_from_llamafactory_generated_predictions()
        self.samples = len(raw_data)
        for id, sample in enumerate(raw_data):
            self.parse_single_generated_predictions(sample, id)
        metrics = self.compute_metrics_two_class()
        print(json.dumps(metrics, indent=4))
        with open(self.saved_data_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        os.chmod(self.saved_data_path, 0o666)  # world-readable/writable output
            



    #print("\nLabel:", data[1]['label'])
    #print("\nPrediction:", data[1]['predict'])







if __name__ == "__main__":
    test_dir = "~generated_predictions.jsonl"
    saved_dir =  "~AD_task_results.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=test_dir)
    parser.add_argument("--output_dir", type=str, default=saved_dir)
    parser.add_argument("--mode", type=str, default="AD-Task")

    args = parser.parse_args()


    print('Arguments: ', args)
    AD_Task_Evaluation_obj = AD_Task_Evaluation(args.input_dir, args.output_dir)
    AD_Task_Evaluation_obj.run()    
