from datasets import load_dataset

import common_util
id2label_data=common_util.read_json("/home/anyang/projects/yuanmei/parts_hf/index.json")
id2label={}
for k,v in id2label_data.items():
    id2label[int(k)]=v

label2id={}
for k,v in id2label.items():
    label2id[v]=k

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    # ds = load_dataset(name, split=split)
    # ds = load_dataset(name)
    ds = load_dataset(
            "imagefolder",
            data_dir=name
    )
    ds = ds['train'].train_test_split(test_size=0.1,seed=1)
    return ds

# data=hf_dataset("/opt/tiger/sleep/bcy_dev")
ds = hf_dataset("/home/anyang/projects/yuanmei/parts_hf")
# ds = load_dataset('beans')

from transformers import ViTFeatureExtractor
# model_name_or_path = 'google/vit-base-patch16-224-in21k'
model_name_or_path = "/home/anyang/projects/yuanmei/classification/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    labels=example['labels']

    label_ids=[]
    for i in list(labels):
        label_ids.append(label2id[i])

    # inputs['labels'] = example['labels']
    inputs['labels'] = label_ids[:1]
    return inputs


# ds = load_dataset('beans')


def transform(example_batch):
    for index,i in enumerate(example_batch['image']):
        if i.mode=="RGBA":
            # 转换为RGB
            import PIL
            rgb_image = PIL.Image.new("RGB", i.size, (255, 255, 255))
            rgb_image.paste(i,mask=i.split()[3])
            example_batch['image'][index]=rgb_image
        elif i.mode=="P":
            from PIL import Image
            rgb_image=i.convert("RGB")
            example_batch['image'][index]=rgb_image
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    
    import numpy as np
    for index,i in enumerate(example_batch['labels']):
        label_ids=[0]*len(label2id)
        for i in list(i):
            # label_ids.append(int(label2id[i]))
            label_ids[int(label2id[i])]=1.0

        
        # label_ids=label_ids[:1]
        example_batch['labels'][index]=label_ids
        # print(i)

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


prepared_ds = ds.with_transform(transform)

import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


import numpy as np
from datasets import load_metric

metric = load_metric(path="./hf_config/accurac.py")
def compute_metrics(p):
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # 防止指数溢出
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    softmax_pred = softmax(p.predictions)
    labels = np.argmax(softmax_pred, axis=1)
    probabilities = np.max(softmax_pred, axis=1)
    selected_labels = labels[probabilities > 0.5]

    labels=set()
    for index,i in enumerate(p.label_ids):
        labels.update([int(index)])

    acc=0
    total=0
    for i in selected_labels:
        if i in labels:
            acc+=1

    for i in p.label_ids:
        total+=sum(i)

    # predictions=
    # return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    # return metric.compute(predictions=selected_labels, references=p.label_ids)
    return {
        "acc":acc/total
    }


from transformers import ViTForImageClassification



# labels = ds['train'].features['labels'].names
# labels=["jiagu","jin","xiaozhuan"]
# print(len(labels))

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(label2id),
    # id2label={str(i): c for i, c in enumerate(labels)},
    # label2id={c: str(i) for i, c in enumerate(labels)},
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification"
)


from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./anime_classification_quality",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=50,
  fp16=True,
  save_steps=500,
  eval_steps=500,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=feature_extractor,
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
