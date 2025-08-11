import dice_ml
from dice_ml.utils import helpers 
from sklearn.model_selection import train_test_split
import json
import os

RAG_CFE_DIR = "/cf_pairs"

dataset = helpers.load_adult_income_dataset()
target = dataset["income"] 
train_dataset, test_dataset, _, _ = train_test_split(dataset,
                                                     target,
                                                     test_size=0.2,
                                                     random_state=0,
                                                     stratify=target)
# Dataset for training an ML model
d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=['age', 'hours_per_week'],
                 outcome_name='income')

# Pre-trained ML model
m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
                  backend='TF2', func="ohe-min-max")
# DiCE explanation instance
exp = dice_ml.Dice(d,m)

query_instance = test_dataset.drop(columns="income")
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=5, desired_class="opposite")

data_json = dice_exp.to_json()
data = json.loads(data_json) 

test_data = data["test_data"]
cfs_list = data["cfs_list"]

output_dir = RAG_CFE_DIR
os.makedirs(output_dir, exist_ok=True)

for idx, test_point in enumerate(test_data):
    features = test_point[0]
    income = features[-1]  
    
    if income == 0:
        filtered_entry = {
            "test_point": test_point,
            "counterfactuals": cfs_list[idx]
        }
        
        filename = f"test_point_{idx}.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(filtered_entry, f, indent=2)

print(f"Saved filtered JSONs in '{output_dir}'")
