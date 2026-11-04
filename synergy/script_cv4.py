from synergy import CCSynergy
import wandb
import os


wandb.login()  # reads the WANDB_API_KEY environment variable or a previous `wandb login`

DIR="" #Working directory (Full Path)
DATAset="Sanger" #Dataset to be used (e.g. "Merck" or "Sanger")
CELLindex=5    #Index for cell line representation method (1-5)
DRUGindex=0    #Index for drug representation method (0-24)
TASK=2  #Regression (1) or Classification (2) task
FOLDnumber=1 #an integer ranginging between 1 and 5
CVtype=4 # selects Data/Cross_Validation_{CVtype}_F{fold}/ in synergy.py
n1=2000 #number of neurons in the first layer - original: 2000
n2=1000 #number of neurons in the second layer - original: 1000
n3=500  #number of neurons in the third layer - o
lr=0.0001 #learning rate
batch=128 #batch size
num=356   #size of the input vector

runs_names = ["25in1-100-unscaled-0.0", "25in1-100-unscaled-0.5", 
              "25in1-100-unscaled-1.0", "25in1-100-unscaled-1.5", "25in1-100-unscaled-2.0", "25in1-100-unscaled-5.0", 
              "25in1-100-scaled-0.0", "25in1-100-scaled-0.5", "25in1-100-scaled-1.0", "25in1-100-scaled-1.5", 
              "25in1-100-scaled-2.0", "25in1-100-scaled-5.0"] # Just names for the runs, in order of the kl_weights list
runs_names = ["PCA"]
runs_names = ["25in1-full-b256"]
runs_names = ["drugzip-25in1-full-0.0-b256"]
# runs_names = ["ae-25in1-full-b256"]
runs_names = ["OHE"]
runs_names = ["MF1024"]
runs_names = ["drugzip-25in1-full-1.0-b256"]
runs_names = ["pca-25in1-full"]


runs = [0] # Just a list of integers to loop over for the runs
for run_n in runs:
    cells = [5]
    for cell in cells:
        CELLindex = cell
        # DEFINING A GROUP ID
        # This tells WandB: "These next 5 runs belong to the same experiment"
        run_name = runs_names[run_n]
        experiment_group = f"{run_name}_Cell_{cell}_CV{CVtype}"
        print("Running for CELLindex =", CELLindex)
        for i in range(1, 6):
            FOLDnumber = i
            print("Running for FOLDnumber =", FOLDnumber)
            run = wandb.init(
                project="CCSynergy-Paper",
                group=experiment_group,  # Groups the 5 folds together
                job_type="fold",         # Tags this run as a 'fold'
                name=f"{experiment_group}_fold_{i}",

                dir=os.environ.get("TMPDIR", "/tmp"),
                
                # TRACK EVERYTHING HERE
                config={
                    "CV_type": CVtype,       
                    "run_n": run_name,           # The specific VAE model used
                    "cell_method": cell,     # Cell line representation
                    "fold": FOLDnumber,
                    "layers": [n1, n2, n3],
                    "learning_rate": lr,
                    "batch_size": batch,
                    "input_size": num,
                    "task": "Classification",
                    "dataset": DATAset
                },
                reinit=True # Allows multiple runs in the same script
            )
            metrics = CCSynergy(str(DIR),str(DATAset),int(CELLindex),int(run_n),int(n1),int(n2),int(n3),int(batch),float(lr),int(num),int(TASK),int(FOLDnumber),int(CVtype),runs_names)
            run.finish()