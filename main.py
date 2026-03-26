#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from utils import *
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY]      = df[Config.TICKET_SUMMARY].astype('U')
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame, target_col: str = None):
    return Data(X, df, target_col=target_col)

# def perform_modelling(data: Data, df: pd.DataFrame, name):
#     model_predict(data, df, name)

def run_chained_multioutput(X: np.ndarray, df: pd.DataFrame):
    
    print("\n" + "#"*60)
    print("  DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT CLASSIFICATION")
    print("#"*60)
 
    df_chained = build_chain_labels(df)
 
    chain_levels = [
        (Config.CHAIN_Y2,       "Chain Level 1: Type2"),
        (Config.CHAIN_Y2_Y3,    "Chain Level 2: Type2 + Type3"),
        (Config.CHAIN_Y2_Y3_Y4, "Chain Level 3: Type2 + Type3 + Type4"),
    ]
 
    all_results = {}
    for col, label in chain_levels:
        data = get_data_object(X, df_chained, target_col=col)
        results = model_predict(data, df_chained, label)
        all_results[label] = results
 
    return all_results

# Code will start executing from following line
if __name__ == '__main__':
    
    # 1. Load & preprocess
    df = load_data()
    df = preprocess_data(df)
 
    # 2. TF-IDF embeddings
    X, df = get_embeddings(df)
 
    # 3. Run chained multi-output across all models (includes Type2-only as Level 1)
    all_results = run_chained_multioutput(X, df)
 
    # 4. Print the full comparison table
    print_comparison_table(all_results)