from transformers import DataProcessor, InputExample
from src.processors import TextClassificationProcessor, processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, text_classification_metrics


class FinancialsClassification(TextClassificationProcessor):
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(InputExample(guid=guid, text_a=line[1] + '. ' + line[2], short_text=line[1] + ".", label=line[0]))
            elif self.task_name == "yelp_review_full":
                examples.append(InputExample(guid=guid, text_a=line[1], short_text=line[1], label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += ' ' + line[2]
                if not pd.isna(line[3]):
                    text += ' ' + line[3]
                examples.append(InputExample(guid=guid, text_a=text, short_text=line[1], label=line[0])) 
            elif self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa', 'funding', 'ma', 'financials', 'partnership']:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        elif self.task_name == "funding":
            return list(range(2))
        elif self.task_name == "ma":
            return list(range(2)) 
        elif self.task_name == "partnership":
            return list(range(2))
        elif self.task_name == "financials":
            return list(range(2)) 
        else:
            raise Exception("task_name not supported.")



processors_mapping['funding'] = FinancialsClassification('funding')
num_labels_mapping['funding'] = 2
output_modes_mapping['funding'] = "classification"
compute_metrics_mapping['funding'] = text_classification_metrics

processors_mapping['financials'] = FinancialsClassification('financials')
num_labels_mapping['financials'] = 2
output_modes_mapping['financials'] = "classification"
compute_metrics_mapping['financials'] = text_classification_metrics

processors_mapping['ma'] = FinancialsClassification('ma')
num_labels_mapping['ma'] = 2
output_modes_mapping['ma'] = "classification"
compute_metrics_mapping['ma'] = text_classification_metrics

processors_mapping['partnership'] = FinancialsClassification('partnership')
num_labels_mapping['partnership'] = 2
output_modes_mapping['partnership'] = "classification"
compute_metrics_mapping['partnership'] = text_classification_metrics