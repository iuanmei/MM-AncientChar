from datasets import load_dataset


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
    # image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    # image_transforms.extend([transforms.ToTensor(),
    #                             transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    # tform = transforms.Compose(image_transforms)

    # assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    # assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    # def pre_process(examples):
    #     processed = {}
    #     processed[image_key] = [tform(im) for im in examples[image_column]]
    #     processed[caption_key] = examples[text_column]
    #     return processed

    # ds.set_transform(pre_process)
    return ds

# data=hf_dataset("/opt/tiger/sleep/bcy_dev")
ds = hf_dataset("/home/anyang/projects/classification/img_datas")
# ds = load_dataset('beans')

from transformers import ViTFeatureExtractor
# model_name_or_path = 'google/vit-base-patch16-224-in21k'
model_name_or_path = "/home/anyang/projects/classification/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
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
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

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
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


from transformers import ViTForImageClassification

# labels = ds['train'].features['labels'].names
labels=["jiagu","jin","xiaozhuan"]
print(len(labels))

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
)


from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./anime_classification_quality",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=20,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
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
