# IGA-An-Intent-Guided-Authoring-Assistant

## Data

The data for fine-tuning IGA can be found [here](https://drive.google.com/drive/folders/1wktlVycCjBCUzBCSWExTqb-d_mbSTmso?usp=sharing).

### Data format explained

Each example consists of two segments seperated by a special token `<sep>`. The segment before `<sep>` contains the masked sentence. The segment after `<sep>` contains the answers that should infill the masked spans in the first segment. 

For instance, an example in the dataset heuristically labeled with the intent of `cause` is shown as below:
```
	... and I wanted to tell you that English is a good language <cause> 
	<sep> because it ’ s easier to learn . <answer> <|endoftext|>
```
The part before `<sep>` is the masked sentence, where the span flagging the writing intent is replaced with a single special token `cause`. This part, most of the time, contains two sentences, the first of which is unmasked and serves as the context for the second masked sentence. When the context does not exist, this sentence is replaced with `<na>`. 

The paraphrase writing intent is the only substitution-based writing intent and uses a slightly different format: the sentence to be paraphrased is enclosed at both sides with the special token `<sub>` and the latter part is the same as described previously. For more details, we refer users to section 4 in our [paper](https://arxiv.org/pdf/2104.07000.pdf).

### Data stats

The training data contains \~1.2M fine-tuning examples covering the following seven writing intents:

| Intent Type  |  Intent Tag  | # of examples |
|--------------|--------------|--------------| 
| Cause		| \<cause\> |  200000 |
| Effect 	| \<effect\> | 108328 |
| Contrast  | \<concession\> | 200000 |
| Description | \<description\> | 198760 |
| Biography | \<biography\> | 2000000 |
| Idiom | \<idiom\> | 1767221 |
| Paraphrase | \<sub\> | 148621 |


## Model

Please download IGA checkpoint from [here](https://drive.google.com/file/d/1D5OzwqKKO_X-80Ba4py55ifT1RbF_55P/view?usp=sharing).

### Command line demo

- Download the model from the link above and unzip
- Run the `generate.py` file with the following command
```
python generate.py --model-path $folder_of_model
```
An example output is shown below:
![Example](https://github.com/SimengSun/IGA-An-Intent-Guided-Authoring-Assistant/image.png)
