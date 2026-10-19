import pandas as pd
import numpy as np
import os
import random
import sys
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, Activation,Flatten
from sklearn.metrics import average_precision_score, mean_squared_error
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import models, layers
from sklearn.metrics import r2_score,mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from sklearn.metrics import mean_squared_error
import wandb
from wandb.integration.keras import WandbCallback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import shap

# Set seed for reproducibility
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def Prepare_Features_Baseline(DIR,DATAset,DRUGindex,CELLindex):
    """
        Prepares features using all 25 signatures, concatenated with 
        Cell Line features.
    
        Args:
            DIR (str): Full path to the working directory.
            DATAset (str): Dataset name (e.g., "Sanger").
            DRUGindex (int): Index for drug representation method (0-24).
            CELLindex (int): Cell line index.
        
        Returns:
            np.ndarray: The concatenated feature matrix.
    """
    FeatureName=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5']
    data=pd.read_csv(str(DIR)+"Data/Synergy_Data/"+str(DATAset)+".csv")
    cell_features=pd.read_csv(str(DIR)+"Data/Cell_Representation/Cell"+str(CELLindex)+".csv")
    drug_features=np.array(pd.read_csv(str(DIR)+"Data/Drug_Representation/"+str(DATAset)+"/"+FeatureName[DRUGindex]+".csv"))
    DD=np.array(data).shape[0]
    features=np.zeros(((2*DD),356))
    data.Cell_Line=data.Cell_Line.astype(str)
    DF=drug_features[:,1:]
    for i in range(DD):
        d1=data.Index1[i]
        d2=data.Index2[i]
        cell=str(data.Cell_Line[i]).strip() 
        features[i,:]=np.concatenate((DF[d1-1,:],DF[d2-1,:],np.array(cell_features[cell])),axis=0)#A-B orientation
        features[(i+DD),:]=np.concatenate((DF[d2-1,:],DF[d1-1,:],np.array(cell_features[cell])),axis=0)#B-A orientation
        print(i)
    return features

def Prepare_Features_Concatenated(DIR,DATAset,CELLindex):
    """
        Prepares features using provided features, concatenated with 
        Cell Line features.
    
        Args:
            DIR (str): Full path to the working directory.
            DATAset (str): Dataset name (e.g., "Sanger").
            CELLindex (int): Cell line index.
        
        Returns:
            np.ndarray: The concatenated feature matrix.
    """
    data=pd.read_csv(str(DIR)+"Data/Synergy_Data/"+str(DATAset)+".csv")
    cell_features=pd.read_csv(str(DIR)+"Data/Cell_Representation/Cell"+str(CELLindex)+".csv") #panda automatically takes header out
    drug_features_filename = "25in1-full-b256"

    # -------
    # If your input array only has the 62 drugs use this :
    # drug_features=np.array(pd.read_csv(str(DIR)+drug_features_filename+".csv", header=0)) # Only take the first 62 rows (the drug features) and all columns (the latent dimensions)
    # DF=drug_features[0:62,:] 

    # If you need to provide the location of the drugs according to the full dataset:
    df_drugs = pd.read_csv(str(DIR) + drug_features_filename + ".csv", header=0)
    drugs_location = [267537, 949222, 1201817, 901593, 828987, 1011120, 390035, 
                      357352, 498965, 1111233, 540645, 299349, 1012489, 554580, 
                      494590, 62047, 433906, 1183005, 1187828, 599545, 382443, 
                      782, 397699, 870011, 1158463, 969916, 179507, 1112338, 
                      55299, 1125014, 730370, 614924, 72201, 947722, 429319, 
                      320080, 406956, 210068, 440410, 816526, 14624, 197492, 
                      206025, 522424, 827414, 1106617, 315251, 162848, 411293, 
                      1040951, 310322, 579991, 60295, 80488, 258357, 524698, 
                      46568, 539191, 255728, 1027781, 182087, 672435]
    drug_features = df_drugs.iloc[drugs_location, :]
    DF = drug_features.values
    # -------

    DD=np.array(data).shape[0]
    features=np.zeros(((2*DD),356)) # Second dimension is d1 + d2 + cell
    data.Cell_Line=data.Cell_Line.astype(str)
    
    for i in range(DD):
        d1=data.Index1[i]
        d2=data.Index2[i]
        cell=str(data.Cell_Line[i]).strip() 
        features[i,:]=np.concatenate((DF[d1-1,:],DF[d2-1,:],np.array(cell_features[cell])),axis=0)#A-B orientation
        features[(i+DD),:]=np.concatenate((DF[d2-1,:],DF[d1-1,:],np.array(cell_features[cell])),axis=0)#B-A orientation
        print(i)
    return features

def Prepare_OHE_Features(DIR, DATAset, CELLindex):
    """
    Prepares features using One-Hot Encoding for drug pairs, concatenated with 
    Cell Line features.

    Args:
        DIR (str): Full path to the working directory.
        DATAset (str): Dataset name (e.g., "Sanger").
        CELLindex (int): Index for cell line representation method (1-5).
    
    Returns:
        np.ndarray: The concatenated feature matrix.
    """
    # Load main data and cell features
    data = pd.read_csv(str(DIR) + "Data/Synergy_Data/" + str(DATAset) + ".csv")
    cell_features = pd.read_csv(str(DIR) + "Data/Cell_Representation/Cell" + str(CELLindex) + ".csv")
    
    DD = np.array(data).shape[0]
    data.Cell_Line = data.Cell_Line.astype(str)
    
    # 1. Determine the total number of unique drugs (D_max)
    D_max = max(data['Index1'].max(), data['Index2'].max())
    # OHE size: D_max for Drug 1 + D_max for Drug 2
    OHE_size = 2 * D_max
    
    # Get Cell Feature size (assuming 100 features from the file structure)
    Cell_Feature_size = cell_features.shape[0] # Assuming cell_features rows are features, columns are cell lines
    
    # Total combined feature vector size
    Total_size = OHE_size + Cell_Feature_size

    # Initialize the final feature matrix
    features = np.zeros(((2 * DD), Total_size))

    for i in range(DD):
        # Drug indices are 1-based in the data, so we use d-1 for 0-based indexing
        d1 = data.Index1[i]
        d2 = data.Index2[i]
        cell = str(data.Cell_Line[i]).strip()

        # 2. Create the One-Hot Encoding vector
        ohe_vector = np.zeros(OHE_size)
        
        # OHE for Drug 1 (first D_max positions)
        ohe_vector[d1 - 1] = 1 
        
        # OHE for Drug 2 (second D_max positions, offset by D_max)
        ohe_vector[(d2 - 1) + D_max] = 1 
        
        # 3. Get Cell Line Feature Vector
        cell_vector = np.array(cell_features[cell])
        
        # 4. Concatenate and store (A-B orientation)
        features[i, :] = np.concatenate((ohe_vector, cell_vector), axis=0) 
        
        # 5. Concatenate and store (B-A orientation - reverse order of OHE blocks)
        ohe_vector_rev = np.zeros(OHE_size)
        ohe_vector_rev[d2 - 1] = 1 # d2 in D1 block
        ohe_vector_rev[(d1 - 1) + D_max] = 1 # d1 in D2 block
        
        features[(i + DD), :] = np.concatenate((ohe_vector_rev, cell_vector), axis=0) 
        
    return features

def Synergy_Scores(DIR,DATAset):
    """
    Retrieves synergy scores from the specified dataset and prepares a response vector.
    
        Args:
            DIR (str): Full path to the working directory.
            DATAset (str): Dataset name (e.g., "Sanger").
        
        Returns:
            np.ndarray: The response "Y" vector.
    """
    data=pd.read_csv(str(DIR)+"Data/Synergy_Data/"+str(DATAset)+".csv")
    DD=np.array(data).shape[0]
    responses=np.zeros(((2*DD)))
    Synergy=data['Synergy']
    for i in range(DD):
        responses[i]=Synergy[i]#A-B orientation
        responses[(i+DD)]=Synergy[i]#B-A orientation      
    return responses

def DNN(inputLength,n1=2000,n2=1000,n3=500,lr=0.0001,TASK=1):
    """
        Creates a Deep Neural Network model for synergy prediction.
    
        Args:
            inputLength (int): Length of the input feature vector.
            n1 (int): Number of neurons in the first layer.
            n2 (int): Number of neurons in the second layer.
            n3 (int): Number of neurons in the third layer.
            lr (float): Learning rate for the optimizer.
            TASK (int): Task type, 1 for Regression and 2 for Classification.
        
        Returns:
            keras.Model: Compiled Keras model ready for training.
    """
    # Added for reproducibility
    initializer = tf.keras.initializers.HeNormal(seed=SEED)

    model = models.Sequential()
    model.add(layers.Dense(n1,kernel_initializer=initializer, input_shape=[inputLength]))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n2, activation='relu',kernel_initializer=initializer))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n3, activation='tanh',kernel_initializer=initializer))
    if TASK==1: #Regression
       model.add(layers.Dense(1))
       model.compile( optimizer=keras.optimizers.Adam(learning_rate=float(lr),beta_1=0.9, beta_2=0.999, amsgrad=False), loss='mean_squared_error',metrics=['mse', 'mae'])
    else: #Classification
       model.add(layers.Dense(1,activation = "sigmoid")) 
       model.compile( optimizer=keras.optimizers.Adam(learning_rate=float(lr),beta_1=0.9, beta_2=0.999, amsgrad=False), loss = "binary_crossentropy", metrics=["accuracy"])
    return model

def CCSynergy(DIR,DATAset,CELLindex,DRUGindex,n1,n2,n3,batch,lr,num,TASK,FOLDnumber,CVtype, FeatureVecorName=["drugzip-25in1-full-b256"]):
    """
        Performs synergy prediction using a Deep Neural Network model.
    
        Args:
            DIR (str): Full path to the working directory.
            DATAset (str): Dataset name (e.g., "Sanger").
            CELLindex (int): Index for cell line representation method (1-5).
            DRUGindex (int): Index for drug representation method (0-24).
            n1 (int): Number of neurons in the first layer.
            n2 (int): Number of neurons in the second layer.
            n3 (int): Number of neurons in the third layer.
            batch (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
            num (int): Size of the input feature vector.
            TASK (int): Task type, 1 for Regression and 2 for Classification.
            FOLDnumber (int): Fold number for cross-validation (1-5).
            CVtype (int): Cross-validation type, 1 for CV1, 2 for CV2, and 4 for CV4.
            FeatureVecorName (list): List of feature vector names, defaulting to ["drugzip-25in1-full-b256"].
        
        Returns:
            tf.keras.Model: The trained Deep Neural Network model.
    """
    # Reading synergy data and indices
    data=pd.read_csv(str(DIR)+"Data/Synergy_Data/"+str(DATAset)+".csv")    
    Training=pd.read_csv(str(DIR)+f"Data/Cross_Validation_{CVtype}_F"+str(FOLDnumber)+"/Training_index.csv")## Needs to be modified depending on Cross validation scheme, synergy dataset and the testing fold
    Testing=pd.read_csv(str(DIR)+f"Data/Cross_Validation_{CVtype}_F"+str(FOLDnumber)+"/Testing_index.csv")## Needs to be modified depending on Cross validation scheme, synergy dataset and the testing fold
    Validation=pd.read_csv(str(DIR)+f"Data/Cross_Validation_{CVtype}_F"+str(FOLDnumber)+"/Validation_index.csv")## Needs to be modified depending on Cross validation scheme, synergy dataset and the testing fold
    FeatureName = FeatureVecorName  # List of feature vector names, defaulting to ["drugzip-25in1-full-b256"]
    # Choose and uncomment one of the following feature preparation methods based on your needs:
    # X = Prepare_Features_Baseline(DIR,DATAset,DRUGindex,CELLindex)
    X = Prepare_Features_Concatenated(DIR, DATAset, CELLindex)
    # X = Prepare_OHE_Features(DIR, DATAset, CELLindex)

    num = X.shape[1]
    Y=Synergy_Scores(DIR,DATAset)
    #Train, Test and Validation data
    X_train=X[Training.Index]
    Y_train=Y[Training.Index] 
    Y_train=np.array(Y_train)        
    X_test=X[Testing.Index]
    Y_test=Y[Testing.Index]    
    X_val=X[Validation.Index]
    Y_val=Y[Validation.Index]
    #Model training and prediction
    DNN_model=DNN(num,n1,n2,n3,lr,TASK)
    cb_check = ModelCheckpoint((str(DIR)+'Results_'+str(DATAset)+'_CCSynergy'+str(CELLindex)+'_'+str(FeatureName[DRUGindex])+'_Fold'+str(FOLDnumber)+'_CV'+str(CVtype)+'.keras'), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    wandb_cb = WandbCallback(save_model=False)
    DNN_model.fit(x=X_train,y=Y_train,batch_size=batch,epochs = 1000,shuffle=True,validation_data = (X_val,Y_val),callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = 10),cb_check,wandb_cb] )
    DNN_model = tf.keras.models.load_model((str(DIR)+'Results_'+str(DATAset)+'_CCSynergy'+str(CELLindex)+'_'+str(FeatureName[DRUGindex])+'_Fold'+str(FOLDnumber)+'_CV'+str(CVtype)+'.keras'))
    predict_test=DNN_model.predict(X_test)

    
    # =====================================================================
    # SHAP ANALYSIS: Testing the "Cell Line Dominance" Hypothesis
    # =====================================================================
    print("\n--- Running SHAP Analysis ---")
    
    # 1. SHAP needs a "background" dataset to understand the baseline. 
    # We take 500 random samples from your training data to save compute time.
    np.random.seed(42)
    background_idx = np.random.choice(X_train.shape[0], min(500, X_train.shape[0]), replace=False)
    background_data = X_train[background_idx]
    
    # 2. We select 200 random samples from the test set to explain.
    test_idx = np.random.choice(X_test.shape[0], min(200, X_test.shape[0]), replace=False)
    test_data = X_test[test_idx]
    
    # 3. Initialize the SHAP DeepExplainer
    # (Note: If DeepExplainer throws a TF2 error, change it to shap.GradientExplainer)
    explainer = shap.DeepExplainer(DNN_model, background_data)
    
    # 4. Calculate SHAP values for the test data
    shap_values = explainer.shap_values(test_data)
    
    # In Keras regression, shap_values might be wrapped in a list. We extract the array.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # 5. Calculate the absolute average importance for all 356 features
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 6. Group the feature importances into the three "Buckets"
    # Features 0 to 127 = Drug A
    # Features 128 to 255 = Drug B
    # Features 256 to 355 = Cell Line
    drug_a_importance = np.sum(mean_abs_shap[0:128])
    drug_b_importance = np.sum(mean_abs_shap[128:256])
    cell_line_importance = np.sum(mean_abs_shap[256:356])
    
    total_importance = drug_a_importance + drug_b_importance + cell_line_importance
    
    # 7. Calculate the percentage of decision-making power each bucket holds
    pct_a = (drug_a_importance / total_importance) * 100
    pct_b = (drug_b_importance / total_importance) * 100
    pct_cell = (cell_line_importance / total_importance) * 100
    
    print("\n=======================================================")
    print("SHAP FEATURE IMPORTANCE BUCKETS")
    print("=======================================================")
    print(f"Drug A Features (0-127)   : {pct_a:.2f}% of model focus")
    print(f"Drug B Features (128-255) : {pct_b:.2f}% of model focus")
    print(f"Cell Line Features(256-355): {pct_cell:.2f}% of model focus")
    print("=======================================================\n")
    # =====================================================================

    auc_score = roc_auc_score(Y_test, predict_test)
    pr_auc_score = average_precision_score(Y_test, predict_test)
    acc_score = accuracy_score(Y_test, np.round(predict_test))
    f1 = f1_score(Y_test, np.round(predict_test))
    print(f"Fold Results -> Accuracy: {acc_score:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}, PR AUC: {pr_auc_score:.4f}")    
    wandb.log({
        "test_auc": auc_score, 
        "test_accuracy": acc_score,
        "test_f1": f1,
        "pr_auc": pr_auc_score
    })

    #Final output
    Final=pd.DataFrame()
    Final['Real']=Y_test     
    Final['Prediction']=predict_test
    print("Results are saved in the working directory.")
    Final.to_csv((str(DIR)+'Results_'+str(DATAset)+'_CCSynergy'+str(CELLindex)+'_'+str(FeatureName[DRUGindex])+'_Fold'+str(FOLDnumber)+'_CV'+str(CVtype)+'.csv'))## The final output file 
 
def main():
    # Use main or call CCSynergy() function directly with the required parameters
    print("CCSynergy Program Started")
    ## Parameters and their default values
    DIR="" #Working directory (Full Path)
    DATAset="Sanger" #Dataset to be used (e.g. "Merck" or "Sanger")
    CELLindex=5    #Index for cell line representation method (1-5)
    DRUGindex=1    #Index for drug representation method (0-24)
    TASK=2  #Regression (1) or Classification (2) task
    FOLDnumber=1 #an integer ranginging between 1 and 5
    CVtype=1 #CV1 (1) or CV2 (2) or CV4 (4)
    n1=2000 #number of neurons in the first layer - original: 2000
    n2=1000 #number of neurons in the second layer - original: 1000
    n3=500  #number of neurons in the third layer - o
    lr=0.0001 #learning rate
    batch=128 #batch size
    num=356   #size of the input vector
    # Feature vector name (default: "drugzip-25in1-full-b256"), 
    # if reproducing CCSynergy, provide [A1, A2, ..., E5] 
    # and uncomment the Prepare_Features_Baseline() function in CCSynergy() 
    # to use the 25 individual signatures instead of the concatenated feature vector.
    FeatureVecorName=["drugzip-25in1-full-b256"] 
    ## Running the software program
    CCSynergy(str(DIR),str(DATAset),int(CELLindex),int(DRUGindex),int(n1),int(n2),int(n3),int(batch),float(lr),int(num),int(TASK),int(FOLDnumber),int(CVtype), FeatureVecorName)


# main()


