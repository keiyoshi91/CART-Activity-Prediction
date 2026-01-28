import logging
from collections import OrderedDict

import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

logger = logging.getLogger(__name__)


class PretrainedEsmModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.load()

    def load(self) -> None:
        self.config = EsmConfig.from_pretrained(self.model_name, output_attentions=True)
        self.config.update({"output_hidden_states": True})
        self.tokenizer = EsmTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=False,
        )
        self.model = EsmModel.from_pretrained(self.model_name, config=self.config)
        self.masked_LM = EsmForMaskedLM.from_pretrained(
            self.model_name, config=self.config
        )


class FinetunedEsmModel(PretrainedEsmModel):
    def __init__(
        self,
        model_name,
        checkpoint_path: str,
        finetuned_model_type="mlm",
        ignore_suffix="lora",
        remove_suffix="module.",
    ):
        super().__init__(model_name)
        self.checkpoint_path = checkpoint_path
        self.finetuned_model_type = finetuned_model_type
        self.ignore_suffix = ignore_suffix
        self.remove_suffix = remove_suffix
        self.checkpoint = torch.load(self.checkpoint_path)
        self.update()

    @property
    def epoch(self) -> float:
        return self.checkpoint["epoch"]

    def update(self):
        state_dict = self.checkpoint["model_state_dict"]
        state_dict = self.update_state_dict(
            state_dict, self.ignore_suffix, self.remove_suffix
        )
        try:
            if self.finetuned_model_type == "mlm":
                self.masked_LM.load_state_dict(state_dict, strict=True)
                logger.info("action=update status=model updated")
                print("All keys matched successfully")
        except Exception as e:
            logger.error("action=update error=%s", e)
            raise

    def __str__(self) -> str:
        return f"FinetunedEsmModel('base model={self.model_name}', 'model type={self.finetuned_model_type}', 'epoch={self.epoch}')"

    def update_state_dict(
        self, state_dict: OrderedDict, ignore_suffix: str, remove_suffix: str
    ):
        updated_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if ignore_suffix in k:
                continue
            name = k.replace(remove_suffix, "")
            updated_state_dict[name] = v
        return updated_state_dict
