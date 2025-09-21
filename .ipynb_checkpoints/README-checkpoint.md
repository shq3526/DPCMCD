First, you need to download the corresponding model from [Hugging Face](https://www.hugging-face.org/) and store it in the `./model` folder. Then, modify the model path in the files `fewshot.py` and `openprompt/pipline_base.py`.



Example shell scripts:
```bash
python fewshot.py --result_file ./output_fewshot.txt --dataset cn_clickbait --template_id 0 --seed 123 --shot 10 --verbalizer manual
```
You can also run the entire model by executing `autorun.py`:

```bash
python autorun.py
```
To Run with Your Own Dataset
1. Format your dataset to match the style we provided and split it into training and testing sets.
2. In the `scripts/TextClassification` directory, create the necessary files, including `verbalizer.txt` and `template.txt`, following our examples. Please ensure the categories in the verbalizer.txt file match those in your dataset.
3. Modify `openprompt/data_utils/text_classification_dataset.py` according to the data processing functions we provided.


Our dataset is stored in `/datasets/TextClassification`, but due to the large number of images, we are unable to upload the entire dataset to GitHub. If you need the complete dataset, please contact me via email, and I will send it to you through other means. My email is: `wangye_lj@163.com`