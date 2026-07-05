import json, numpy as np, pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.auto_encoder import AutoEncoder

# --- Data loading ---
def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    X = np.array([item[:-1] for item in data])
    y = np.array([item[-1] for item in data]) # 1 = anomalous, 0 = normal
    return X, y

# load and merge the training sets
source_data_path_train = "./AD_Source_0_to_6_Processed.json"
tesd_data_train = "./TSED_Train.json"
X_train_raw, y_train = load_data(source_data_path_train)
TESD_X_train, TSED_y_train = load_data(tesd_data_train)
X_train = np.concatenate((X_train_raw, TESD_X_train), axis=0)
y_train = np.concatenate((y_train, TSED_y_train), axis=0)

# test set
source_data_path_test = "./AD_Source_8_and_9_Processed.json"
tesd_data_test = "./TSED_Test.json"
X_test, y_test = load_data(source_data_path_test)
TSED_X_test, TSED_y_test = load_data(tesd_data_test)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
TSED_X_test_s = scaler.transform(TSED_X_test)

# compute the contamination ratio
class_counts = np.bincount(y_train.astype(int))
contamination = class_counts[1] / len(y_train)

# --- Shared evaluation logic ---
def calculate_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    # labels=[0,1] guarantees unpacking even if y_pred is all zeros
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return [acc, precision, recall, f1, fpr]

def print_results(name, ans, ans_tsed):
    print(f"\n[{name} Final Average Results]")
    print(f"Standard -> Acc: {ans[0]:.2f}%, Prec: {ans[1]:.2f}%, Rec: {ans[2]:.2f}%, F1: {ans[3]:.2f}%, FPR: {ans[4]:.2f}%")
    print(f"TSED     -> Acc: {ans_tsed[0]:.2f}%, Prec: {ans_tsed[1]:.2f}%, Rec: {ans_tsed[2]:.2f}%, F1: {ans_tsed[3]:.2f}%, FPR: {ans_tsed[4]:.2f}%")

# --- Method 1: Random Forest (supervised) ---
def run_rf(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, round_num=5):
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        model = RandomForestClassifier(n_estimators=100, random_state=i)
        model.fit(X_train, y_train)
        
        # RF outputs 0/1 directly; no conversion needed
        res = calculate_metrics(y_test, model.predict(X_test))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test))
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x/round_num*100 for x in ans]
    ans_tsed = [x/round_num*100 for x in ans_tsed]
    print_results("Random Forest", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 2: Isolation Forest (unsupervised AD) ---
def run_iforest(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, contamination, round_num=5):
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        model = IsolationForest(contamination=contamination, random_state=i)
        model.fit(X_train_s)
        
        # map: -1 (anomalous) -> 1, 1 (normal) -> 0
        y_pred = np.where(model.predict(X_test_s) == -1, 1, 0)
        y_pred_t = np.where(model.predict(TSED_X_test_s) == -1, 1, 0)
        
        res = calculate_metrics(y_test, y_pred)
        res_t = calculate_metrics(TSED_y_test, y_pred_t)
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x/round_num*100 for x in ans]
    ans_tsed = [x/round_num*100 for x in ans_tsed]
    print_results("Isolation Forest", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 3: One-Class SVM (semi-supervised AD) ---
def run_svm(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, round_num=5):
    ans, ans_tsed = [0.0]*5, [0.0]*5
    # OneClassSVM is deterministic; the loop is kept only for a uniform format
    model = OneClassSVM(gamma='auto')
    model.fit(X_train_s)
    
    for i in range(round_num):
        y_pred = np.where(model.predict(X_test_s) == -1, 1, 0)
        y_pred_t = np.where(model.predict(TSED_X_test_s) == -1, 1, 0)
        
        res = calculate_metrics(y_test, y_pred)
        res_t = calculate_metrics(TSED_y_test, y_pred_t)
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x/round_num*100 for x in ans]
    ans_tsed = [x/round_num*100 for x in ans_tsed]
    print_results("OneClassSVM", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 4: LOF (unsupervised AD) ---
def run_lof(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, contamination, round_num=5):
    # novelty=True enables fit-then-predict
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination)
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        y_pred = np.where(model.predict(X_test_s) == -1, 1, 0)
        y_pred_t = np.where(model.predict(TSED_X_test_s) == -1, 1, 0)
        
        res = calculate_metrics(y_test, y_pred)
        res_t = calculate_metrics(TSED_y_test, y_pred_t)
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x/round_num*100 for x in ans]
    ans_tsed = [x/round_num*100 for x in ans_tsed]
    print_results("LOF", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 5: XGBoost (supervised) ---
def run_xgboost(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, round_num=5):
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_s, y_train)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x/round_num*100 for x in ans]
    ans_tsed = [x/round_num*100 for x in ans_tsed]
    print_results("XGBoost", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 6: LightGBM (supervised) ---
def run_lightgbm(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, round_num=5):
    # initialize the LightGBM classifier
    # num_leaves controls model complexity; usually kept within 2^max_depth
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,      # core LightGBM parameter (default 31)
        random_state=42,
        importance_type='gain', # feature-importance criterion
        verbosity=-1        # silence training warnings/logs
    )
    
    # fit the model
    model.fit(X_train_s, y_train)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        # compute metrics with calculate_metrics
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    # print the aggregated results
    print_results("LightGBM", ans, ans_tsed)
    
    return ans, ans_tsed

# --- Method 7: ECOD (unsupervised) ---
def run_ecod(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    # initialize the ECOD model
    # contamination is the anomaly ratio, the key parameter of unsupervised AD
    model = ECOD(contamination=contamination)
    
    # fit the model (unsupervised; only X is needed)
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        # ECOD is deterministic; repeated predictions are identical
        # model.predict returns 0 or 1
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    # print the aggregated results
    print_results("ECOD", ans, ans_tsed)
    
    return ans, ans_tsed

# --- Method 8: COPOD (unsupervised) ---
def run_copod(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    COPOD anomaly detection
    :param contamination: anomaly ratio in (0, 0.5], set from the dataset
    """
    # 1. initialize the model
    # COPOD is distribution-based and deterministic
    model = COPOD(contamination=contamination)
    
    # 2. fit the model
    # note: unsupervised fitting only needs X_train_s to learn the normal distribution
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    
    # 3. evaluate over repeated runs
    for i in range(round_num):
        # model.predict() returns 0/1 labels directly
        # compute metrics with calculate_metrics
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # 4. average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    # 5. print the results
    print_results("COPOD", ans, ans_tsed)
    
    return ans, ans_tsed

# --- Method 9: KNN (unsupervised) ---
def run_knn(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    kNN anomaly detection: scores by the distance to the k-th neighbor
    """
    # initialize the model (n_neighbors defaults to 5; 20 is usually more robust)
    model = KNN(contamination=contamination, n_neighbors=20)
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    print_results("kNN", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 10: HBOS (distribution-based) ---
def run_hbos(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    HBOS anomaly detection: histogram-based scoring, very fast
    """
    model = HBOS(contamination=contamination)
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    print_results("HBOS", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 11: LODA (ensemble) ---
def run_loda(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    LODA anomaly detection: random-projection histograms, suited to sparse high-dimensional data
    """
    model = LODA(contamination=contamination)
    model.fit(X_train_s)
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    for i in range(round_num):
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    print_results("LODA", ans, ans_tsed)
    return ans, ans_tsed


# --- Method 12: AutoEncoder (neural network) ---
def run_autoencoder(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num=5):
    import numpy as np
    from pyod.models.auto_encoder import AutoEncoder
    
    ans, ans_tsed = [0.0]*5, [0.0]*5
    
    # derive the hidden-layer sizes dynamically, e.g. 10 features -> [10, 8, 8, 10]
    n_features = X_train_s.shape[1]
    hidden_neurons = [max(2, n_features // 2), max(2, n_features // 4), max(2, n_features // 2)]

    print("-" * 50)
    for i in range(round_num):
        print(f"AutoEncoder - Round {i+1}/{round_num}")
        
        # 1. instantiate the model
        # note: PyOD's AutoEncoder is a neural model; fix random_state for stability
        model = AutoEncoder(
            hidden_neuron_list=[64, 32, 32, 64],
            epoch_num=20, 
            contamination=contamination, 
            random_state = 42,
            verbose=0
        )
        
        # 2. fit
        model.fit(X_train_s)
        
        # 3. predict (PyOD outputs 0/1 directly)
        y_pred = model.predict(X_test_s)
        y_pred_t = model.predict(TSED_X_test_s)
        
        # 4. compute metrics (calculate_metrics performs no -1 mapping)
        res = calculate_metrics(y_test, y_pred)
        res_t = calculate_metrics(TSED_y_test, y_pred_t)
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    print_results("AutoEncoder", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 13: VAE (unsupervised, PyTorch) ---
def run_vae(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    VAE anomaly detection (Kingma et al., 'Auto-Encoding Variational Bayes')
    """
    import numpy as np
    from pyod.models.vae import VAE
    
    n_features = X_train_s.shape[1]
    
    # follows the documented [feature_size, 128, 64, 32, latent_dim] layout
    # encoder: decreasing widths
    enc_list = [128, 64, 32] if n_features > 128 else [64, 32]
    # decoder: increasing widths (symmetric)
    dec_list = [32, 64, 128] if n_features > 128 else [32, 64]

    ans, ans_tsed = [0.0]*5, [0.0]*5
    print("-" * 50)
    
    for i in range(round_num):
        print(f"VAE - Round {i+1}/{round_num}")
        
        # 1. initialize the model
        # note: parameter names must be encoder_neuron_list / decoder_neuron_list
        model = VAE(
            encoder_neuron_list=enc_list,
            decoder_neuron_list=dec_list,
            latent_dim=2,
            epoch_num=30,           # library default
            contamination=contamination,
            random_state=i,
            verbose=0
        )
        
        # 2. fit the model (unsupervised)
        model.fit(X_train_s)
        
        # 3. predict and evaluate
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    print_results("VAE", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 14: DeepSVDD (unsupervised) ---
def run_deepsvdd(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=5):
    """
    DeepSVDD anomaly detection
    """
    import numpy as np
    from pyod.models.deep_svdd import DeepSVDD
    
    # feature dimensionality (required positional argument)
    n_features = X_train_s.shape[1]
    
    # derive the hidden-layer sizes dynamically
    my_hidden_neurons = [max(2, n_features // 2), max(2, n_features // 4)]

    print("-" * 50)
    ans, ans_tsed = [0.0]*5, [0.0]*5
    
    for i in range(round_num):
        print(f"DeepSVDD - Round {i+1}/{round_num}")
        
        # 1. initialize the model
        model = DeepSVDD(
            n_features=n_features,            # required positional argument
            hidden_neurons=my_hidden_neurons, 
            epochs=50,                        
            use_ae=True,                      
            contamination=contamination, 
            random_state=i, 
            verbose=0
        )
        
        # 2. fit the model
        model.fit(X_train_s)
        
        # 3. predict and evaluate
        res = calculate_metrics(y_test, model.predict(X_test_s))
        res_t = calculate_metrics(TSED_y_test, model.predict(TSED_X_test_s))
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    print_results("DeepSVDD", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 15: Matrix Profile (unsupervised) ---
def run_matrix_profile(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=contamination, round_num=1):
    """
    Matrix Profile (implemented with STUMPY)
    Flags anomalies via the Euclidean distance between each subsequence and its nearest neighbor.
    """
    import numpy as np
    import stumpy
    from sklearn.preprocessing import MinMaxScaler

    # Matrix Profile applies to univariate (or specific multivariate) series
    # for multivariate X, take the first column / a reduced feature as the primary signal
    def get_mp_scores(data, window_size=20):
        # ensure the input is a 1-D array
        if len(data.shape) > 1:
            data_1d = data[:, 0].flatten()
        else:
            data_1d = data.flatten()
            
        # compute the Matrix Profile
        mp = stumpy.stump(data_1d, m=window_size)
        # mp[:, 0] is the distance vector; larger means more anomalous
        scores = mp[:, 0]
        
        # pad the leading window to match the original length
        pad_size = len(data_1d) - len(scores)
        scores = np.pad(scores, (pad_size, 0), mode='edge')
        return scores

    print("-" * 50)
    # Matrix Profile is deterministic; a single run (round_num=1) suffices
    ans, ans_tsed = [0.0]*5, [0.0]*5
    
    # sliding-window size, the key MP parameter
    m = 20 

    for i in range(round_num):
        print(f"Matrix Profile - Processing...")

        # 1. compute anomaly scores
        test_scores = get_mp_scores(X_test_s, window_size=m)
        tsed_test_scores = get_mp_scores(TSED_X_test_s, window_size=m)

        # 2. threshold by the contamination ratio and convert to labels
        threshold = np.percentile(test_scores, 100 * (1 - contamination))
        pred_test = (test_scores > threshold).astype(int)
        
        threshold_t = np.percentile(tsed_test_scores, 100 * (1 - contamination))
        pred_tsed = (tsed_test_scores > threshold_t).astype(int)

        # 3. evaluate
        res = calculate_metrics(y_test, pred_test)
        res_t = calculate_metrics(TSED_y_test, pred_tsed)
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs and convert to percentages
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    print_results("Matrix Profile", ans, ans_tsed)
    return ans, ans_tsed

# --- Method 16: LSTM-ED (reconstruction model from Merlion) ---
def run_lstmen(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination=0.1, round_num=1):
    """
    LSTM Encoder-Decoder anomaly detection
    Captures temporal dependencies with an LSTM and scores anomalies by reconstruction error.
    """
    import numpy as np
    import pandas as pd
    try:
        from merlion.models.anomaly.lstm_ed import LSTMED, LSTMEDConfig
        from merlion.utils.time_series import TimeSeries
    except ImportError:
        print("Error: salesforce-merlion not found; run pip install salesforce-merlion")
        return [0.0]*5, [0.0]*5

    # 1. helper: convert NumPy arrays to Merlion TimeSeries
    def to_merlion(data):
        # LSTM-ED needs a timestamp index; create a synthetic per-second index
        df = pd.DataFrame(data)
        df.index = pd.date_range(start='2026-01-01', periods=len(df), freq='S')
        return TimeSeries.from_pd(df)

    print("-" * 50)
    ans, ans_tsed = [0.0]*5, [0.0]*5
    
    # convert the data format
    train_merlion = to_merlion(X_train_s)
    test_merlion = to_merlion(X_test_s)
    tsed_merlion = to_merlion(TSED_X_test_s)

    for i in range(round_num):
        print(f"LSTMEN - Round {i+1}/{round_num}")
        
        # 2. model configuration
        # sequence_len is the sliding-window size (96 recommended)
        config = LSTMEDConfig(
            sequence_len=96,
            hidden_size=32,
            n_epochs=10,
            batch_size=64,
            lr=1e-3
        )
        model = LSTMED(config)
        
        # 3. fit
        model.train(train_data=train_merlion)
        
        # 4. obtain scores and convert to labels
        # get_anomaly_score returns a per-point anomaly score
        test_scores = model.get_anomaly_score(test_merlion).to_pd().values.flatten()
        tsed_scores = model.get_anomaly_score(tsed_merlion).to_pd().values.flatten()
        
        # 5. threshold by quantile (matching the contamination ratio)
        def get_labels(scores, contam):
            threshold = np.percentile(scores, 100 * (1 - contam))
            return (scores > threshold).astype(int)

        pred_labels = get_labels(test_scores, contamination)
        pred_tsed_labels = get_labels(tsed_scores, contamination)

        # 6. evaluate
        res = calculate_metrics(y_test, pred_labels)
        res_t = calculate_metrics(TSED_y_test, pred_tsed_labels)
        
        for j in range(5):
            ans[j] += res[j]
            ans_tsed[j] += res_t[j]
            
    # average over runs
    ans = [x / round_num * 100 for x in ans]
    ans_tsed = [x / round_num * 100 for x in ans_tsed]
    
    print_results("LSTMEN", ans, ans_tsed)
    return ans, ans_tsed

# --- Run all methods ---
print(f"Data statistics: train size={len(y_train)}, anomaly ratio={contamination:.4%}")
round_num = 5

# run_rf(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, round_num)
# run_iforest(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_svm(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, round_num)
# run_lof(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, contamination, round_num)
# run_xgboost(X_train, y_train, X_test, y_test, TSED_X_test, TSED_y_test, round_num)
# run_lightgbm(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, round_num)
# run_ecod(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_copod(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_knn(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_hbos(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_loda(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_autoencoder(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_vae(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_deepsvdd(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num)
# run_matrix_profile(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num=5)
run_lstmen(X_train_s, y_train, X_test_s, y_test, TSED_X_test_s, TSED_y_test, contamination, round_num=5)