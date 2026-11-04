
gc(full = TRUE)

# Output folder read by synergy.py: Data/Cross_Validation_<CV>_F<fold>[_T<tissue>]
CV_DIR <- function(CV, FOLD, TINDEX = NULL) {
  d <- paste0("Data/Cross_Validation_", CV, "_F", FOLD, if (!is.null(TINDEX)) paste0("_T", TINDEX) else "")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# We introduce two functions for implementing the CV1 and CV2 cross validation schemes as follows:
# 1. CV1 Cross Validation Scheme
CV1_Splitter<-function(DIR,DATAset,SEED,FOLD_TEST){
  ### Arguments
  # DIR: Full path to the directory where the synergy DATA folders locates
  # DATAset: Merck, Sanger, and etc.
  # SEED: see number for random number generation
  # FOLD_TEST: the fold that is considered as the test fold (a number between 1 and 5)
  ### Synergy data and unique drug pairs
  data<-read.csv(paste0("Data/Synergy_Data/",DATAset,".csv"))
  DD<-dim(data)[1]
  U<-unique(data[,c("Index1","Index2")])
  D<-dim(U)[1]
  ### Splitting the unique drug pairs
  require(caret)
  set.seed(SEED)
  S<- createFolds(1:D, k = 5, list = TRUE, returnTrain = FALSE)
  ### Assigning folds to the unique drug pairs 
  TEMP<-numeric(D)
  for (i in 1:5){TEMP[S[[i]]]<-i}
  ### Assigning folds to the samples in the drug synergy data 
  FOLDS<-numeric(DD)
  for (i in 1:D){FOLDS[intersect(which(data[,"Index1"]==U[i,1]),which(data[,"Index2"]==U[i,2]))]<-TEMP[i]}
  ### Generating the Train, Test and Validation Indices
  TEST<-sort(which(FOLDS==FOLD_TEST))
  ALL_TRAIN<-which(FOLDS!=FOLD_TEST)
  set.seed(SEED)
  VAL<-sort(sample(ALL_TRAIN,round(length(ALL_TRAIN)/5)))
  TRAIN<-sort(setdiff(ALL_TRAIN,VAL))
  ### Considering both orientations and applying the (-1) to correct the indices for the Python code
  TEST<-(c(TEST,(TEST+DD))-1)
  VAL<-(c(VAL,(VAL+DD))-1)
  TRAIN<-(c(TRAIN,(TRAIN+DD))-1)
  ### Writing the files, which will be used as inputs in the CCSynergy software
  write.csv(data.frame(Index=TEST),paste0(CV_DIR(1, FOLD_TEST), "/Testing_index.csv"),col.names=TRUE,row.names=FALSE)
  write.csv(data.frame(Index=VAL),paste0(CV_DIR(1, FOLD_TEST), "/Validation_index.csv"),col.names=TRUE,row.names=FALSE)
  write.csv(data.frame(Index=TRAIN),paste0(CV_DIR(1, FOLD_TEST), "/Training_index.csv"),col.names=TRUE,row.names=FALSE)
}


# 2. CV2 Cross Validation Scheme
CV2_Splitter<-function(DIR,DATAset,SEED,FOLD_TEST,TINDEX){
  ### Arguments
  # DIR: Full path to the directory where the synergy DATA folders locates
  # DATAset: Merck, Sanger, and etc.
  # SEED: see number for random number generation
  # FOLD_TEST: the fold that is considered as the test fold (a number between 1 and 5)
  # TINDEX: the index of the given tissue
  ### Synergy data and unique drug pairs
  data<-read.csv(paste0("Data/Synergy_Data/",DATAset,".csv"))
  DD<-dim(data)[1]
  U<-unique(data[,c("Index1","Index2")])
  D<-dim(U)[1]
  ### Splitting the unique drug pairs
  require(caret)
  set.seed(SEED)
  S<- createFolds(1:D, k = 5, list = TRUE, returnTrain = FALSE)
  ### Assigning folds to the unique drug pairs 
  TEMP<-numeric(D)
  for (i in 1:5){TEMP[S[[i]]]<-i}
  ### Assigning folds to the samples in the drug synergy data 
  FOLDS<-numeric(DD)
  for (i in 1:D){FOLDS[intersect(which(data[,"Index1"]==U[i,1]),which(data[,"Index2"]==U[i,2]))]<-TEMP[i]}
  ### Generating the Train, Test and Validation Indices
  TEST<-sort(intersect(which(FOLDS==FOLD_TEST),which(data$Tindex==TINDEX)))
  ALL_TRAIN<-intersect(which(FOLDS!=FOLD_TEST),which(data$Tindex!=TINDEX))
  set.seed(SEED)
  VAL<-sort(sample(ALL_TRAIN,round(length(ALL_TRAIN)/5)))
  TRAIN<-sort(setdiff(ALL_TRAIN,VAL))
  ### Considering both orientations and the applying (-1) to correct the indices for the Python code
  TEST<-(c(TEST,(TEST+DD))-1)
  VAL<-(c(VAL,(VAL+DD))-1)
  TRAIN<-(c(TRAIN,(TRAIN+DD))-1)
  ### Writing the files, which will be used as inputs in the CCSynergy software
  write.csv(data.frame(Index=TEST),paste0(CV_DIR(2, FOLD_TEST, TINDEX), "/Testing_index.csv"),col.names=TRUE,row.names=FALSE)
  write.csv(data.frame(Index=VAL),paste0(CV_DIR(2, FOLD_TEST, TINDEX), "/Validation_index.csv"),col.names=TRUE,row.names=FALSE)
  write.csv(data.frame(Index=TRAIN),paste0(CV_DIR(2, FOLD_TEST, TINDEX), "/Training_index.csv"),col.names=TRUE,row.names=FALSE)
}

#By Mana
CV3_Splitter<-function(DATAset,SEED,DRUG_FOLD_TEST){
  ### Arguments
  # DIR: Full path to the directory
  # DATAset: Sanger, etc.
  # SEED: seed for reproducibility
  # DRUG_FOLD_TEST: the fold of unique drugs to be held out for testing (1-5)

  ### 1. Load Data
  data<-read.csv(paste0("Data/Synergy_Data/",DATAset,".csv"))
  DD<-dim(data)[1]
  
  ### 2. Identify Unique Drugs (Not Pairs)
  all_drugs <- unique(c(data$Index1, data$Index2))
  num_unique_drugs <- length(all_drugs)
  
  ### 3. Split Unique Drugs into 5 Folds
  require(caret)
  set.seed(SEED)
  drug_folds <- createFolds(1:num_unique_drugs, k = 5, list = TRUE, returnTrain = FALSE)
  
  ### 4. Map each drug to a specific fold
  drug_to_fold <- numeric(max(all_drugs))
  for (i in 1:5) {
    drug_to_fold[all_drugs[drug_folds[[i]]]] <- i
  }
  
  ### 5. Define Test Set (Unseen Drugs)
  # A pair is in the TEST set if AT LEAST one of the drugs belongs to the test fold
  test_indices <- which(drug_to_fold[data$Index1] == DRUG_FOLD_TEST | 
                        drug_to_fold[data$Index2] == DRUG_FOLD_TEST)
  
  ### 6. Define Training Pool (Completely Seen Drugs)
  # A pair is only in TRAIN if BOTH drugs are NOT in the test fold
  all_train_indices <- which(drug_to_fold[data$Index1] != DRUG_FOLD_TEST & 
                             drug_to_fold[data$Index2] != DRUG_FOLD_TEST)
  
  ### 7. Create Validation Set from the Training Pool
  set.seed(SEED)
  VAL <- sort(sample(all_train_indices, round(length(all_train_indices)/5)))
  TRAIN <- sort(setdiff(all_train_indices, VAL))
  TEST <- sort(test_indices)
  
  ### 8. Doubling for orientations and Python indexing (-1)
  TEST <- (c(TEST, (TEST + DD)) - 1)
  VAL <- (c(VAL, (VAL + DD)) - 1)
  TRAIN <- (c(TRAIN, (TRAIN + DD)) - 1)
  
  ### 9. Write Files
  write.csv(data.frame(Index=TEST), paste0(CV_DIR(3, DRUG_FOLD_TEST), "/Testing_index.csv"), row.names=FALSE)
  write.csv(data.frame(Index=VAL), paste0(CV_DIR(3, DRUG_FOLD_TEST), "/Validation_index.csv"), row.names=FALSE)
  write.csv(data.frame(Index=TRAIN), paste0(CV_DIR(3, DRUG_FOLD_TEST), "/Training_index.csv"), row.names=FALSE)
}

CV4_Splitter <- function(DATAset, SEED, DRUG_FOLD_TEST){
  ### 1. Load Data
  data <- read.csv(paste0("Data/Synergy_Data/",DATAset,".csv"))
  DD <- dim(data)[1]
  
  ### 2. Identify Unique Drugs
  all_drugs <- unique(c(data$Index1, data$Index2))
  num_unique_drugs <- length(all_drugs)
  
  ### 3. Split Unique Drugs into 5 Folds
  require(caret)
  set.seed(SEED)
  drug_folds <- createFolds(1:num_unique_drugs, k = 5, list = TRUE, returnTrain = FALSE)
  
  ### 4. Map each drug to a specific fold
  drug_to_fold <- numeric(max(all_drugs))
  for (i in 1:5) {
    drug_to_fold[all_drugs[drug_folds[[i]]]] <- i
  }
  
  ### 5. STRICT TEST SET: BOTH drugs must belong to the test fold - Use of AND instead of OR
  test_indices <- which(drug_to_fold[data$Index1] == DRUG_FOLD_TEST & 
                        drug_to_fold[data$Index2] == DRUG_FOLD_TEST)
  
  ### 6. STRICT TRAINING POOL: BOTH drugs must NOT belong to the test fold
  all_train_indices <- which(drug_to_fold[data$Index1] != DRUG_FOLD_TEST & 
                             drug_to_fold[data$Index2] != DRUG_FOLD_TEST)
                             
  # Note: Any pair with one drug in DRUG_FOLD_TEST and one drug outside it 
  # is intentionally ignored to prevent data leakage!

  ### 7. Create Validation Set from the Training Pool (20%)
  set.seed(SEED)
  VAL <- sort(sample(all_train_indices, round(length(all_train_indices)/5)))
  TRAIN <- sort(setdiff(all_train_indices, VAL))
  TEST <- sort(test_indices)
  
  ### 8. Doubling for orientations and Python indexing (-1)
  TEST <- (c(TEST, (TEST + DD)) - 1)
  VAL <- (c(VAL, (VAL + DD)) - 1)
  TRAIN <- (c(TRAIN, (TRAIN + DD)) - 1)
  
  ### 9. Write Output Files
  write.csv(data.frame(Index=TEST), paste0(CV_DIR(4, DRUG_FOLD_TEST), "/Testing_index.csv"), row.names=FALSE)
  write.csv(data.frame(Index=VAL), paste0(CV_DIR(4, DRUG_FOLD_TEST), "/Validation_index.csv"), row.names=FALSE)
  write.csv(data.frame(Index=TRAIN), paste0(CV_DIR(4, DRUG_FOLD_TEST), "/Training_index.csv"), row.names=FALSE)
}

# Generate all five folds of a scheme, e.g.:
# for (f in 1:5) CV1_Splitter("unused", "Sanger", 28, f)
# for (f in 1:5) CV4_Splitter("Sanger", 28, f)

### Example 
# In this example, the index files that are on the GitHub repository are generated as follows
# The cross validation scheme is CV1
# DIR: The working directory
# DATAset: Merck dataset is used
# SEED: we used 28
# FOLD_TEST: 1
# CV1_Splitter("/home/User/CCSynergy","Merck",28,5)
# CV4_Splitter("Merck",28,1)
# CV2_Splitter("/home/User/CCSynergy","Merck",28,1,1)



### First Sanger run 
# CV4_Splitter("Sanger",28,1)
# CV2_Splitter("/home/User/CCSynergy","Sanger",28,1,3)

