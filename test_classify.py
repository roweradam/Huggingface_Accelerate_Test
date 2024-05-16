import torch
import pandas as pd
import transformers
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Value, ClassLabel, Features
import time

output_path = 'accelerate_test/run8_accelerate_4gpu/'
report_string = "SAVSNET HPC Adventures\n"
df =  pd.read_excel("https://www.liverpool.ac.uk/media/livacuk/savsnet/SAVSNET,sample,vet,data.xlsx", engine='openpyxl')
df.to_csv(output_path + "open_dataset.csv")
report_string += f"read {df.shape[0]} lines in\n"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# perform torch rand on gpu
x = torch.rand(5, 3).to(device)

report_string+=("The tensor is:\n")
report_string += str(x)+"\n"


#device_gpu_1 = torch.device("cuda:1")
#device_gpu_0 = torch.device("cuda:0")

#x = torch.rand(5, 3).to(device_gpu_0)
#y = torch.rand(5, 3).to(device_gpu_1)

#print(x)
#print(y)
#exit()

from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

df = pd.read_csv(output_path + "open_dataset.csv")
df = df.dropna(subset=["Narrative", "SAVSNET MPC"]).rename(columns={"Narrative":"text", "SAVSNET MPC":"labels"})
df.to_csv(output_path + "fixed_narratives.csv", index=False)

time_start = time.time()

mpc_features = Features(
    {
       	"text": Value("string"),
       	"labels": ClassLabel(names=list(df["labels"].unique())),

    }
)

dataset = load_dataset(
    "csv",
    data_files={output_path + "fixed_narratives.csv"},
    delimiter=",",
    usecols=["text", "labels"],
    features=mpc_features,
    keep_in_memory=True,
    memory_map=True,
    split="train",
)

#train test split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

config = BertConfig.from_pretrained(
    "bert-base-uncased", num_labels=df["labels"].nunique()
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config
)


def tokenize(batch):
    tokenized_batch = tokenizer(
        batch["text"], padding='max_length', truncation=True, max_length=512
    )
    return tokenized_batch


tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True
)

train_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    output_dir= output_path + "results",
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
time_end = time.time()
print(str(time_end - time_start))

def has_content(filename):
  """
  Checks if a .txt file has any content.

  Args:
      filename: The path to the .txt file.

  Returns:
      True if the file has any content, False otherwise.
  """
  try:
    with open(filename, "r") as f:
      # Read a line (will be empty string if no content)
      line = f.readline().strip()
      return bool(line)  # Check if the line is not empty
  except FileNotFoundError:
    # Handle file not found error (optional)
    return False

# save time_end to a file
if has_content("accelerate_test/time.txt") == False:
    with open("accelerate_test/time.txt", "w") as f:
        f.write(str(time_end - time_start))
else:
    with open("accelerate_test/time.txt", "a") as f:
        f.write('\nRun 8 (Accelerate, 4 GPUs) ' + str(time_end - time_start))

with open(output_path + 'test_result_manual.txt',"w") as result_file:
    result_file.write(report_string)
