import torch

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextGenerator:
    def __init__(self, model_id: str, **kwargs):
        """Args:
            model_id (str): The huggingface model id.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set padding side to left
        self.tokenizer.padding_side = 'left'

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto",
            **kwargs
        )

    def __call__(
        self,
        system_message: str,
        task_que: str,
        instruction: str
    ):
        if len(system_message) == 0:
            model_prompt = f'[INST]\n\n{instruction}[/INST]\n\n{task_que}\n\n'
        else:
            model_prompt = f'[INST] <<SYS>>\n{system_message}'\
                f'\n<</SYS>>\n\n{instruction}[/INST]\n\n{task_que}\n\n'

        # Move stuff to the right device
        text_tokens = self.tokenizer([model_prompt], return_tensors='pt')
        text_tokens = {k: v.to(self.llm.device) for k, v in text_tokens.items()}

        with torch.inference_mode():
            outputs = self.llm.generate(
                **text_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=.95,
                repetition_penalty=1.15,
                max_new_tokens=1444,
                return_dict_in_generate=True
            )

        # Get only the generated token ids.
        input_len = text_tokens['input_ids'].size(1)
        action_ids = outputs.sequences[:, input_len:]

        # Since `outputs` has only one sequence
        output_text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)

        return output_text

class BatchTextGenerator(TextGenerator):
    def __call__(
        self,
        system_message: str,
        task_que: str,
        instructions: List[str],
        return_raw_sequences: bool = False
    ):
        if len(system_message) == 0:
            model_prompts = [f'[INST]\n\n{inst}[/INST]\n\n{task_que}\n\n'
            for inst in instructions
        ]
        else:
            model_prompts = [
            f'[INST] <<SYS>>\n{system_message}'\
                f'\n<</SYS>>\n\n{inst}[/INST]\n\n{task_que}\n\n'
            for inst in instructions
        ]

        # Move stuff to the right device
        text_tokens = self.tokenizer(model_prompts, padding=True, return_tensors='pt')
        text_tokens = {k: v.to(self.llm.device) for k, v in text_tokens.items()}

        with torch.inference_mode():
            outputs = self.llm.generate(
                **text_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=.95,
                repetition_penalty=1.15,
                max_new_tokens=1444,
                return_dict_in_generate=True
            )

        if return_raw_sequences:
            return outputs

        # Get only the generated token ids.
        input_len = text_tokens['input_ids'].size(1)
        gen_ids = outputs.sequences[:, input_len:]

        output_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return output_text
