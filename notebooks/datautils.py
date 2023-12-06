import json
import random
import pathlib
from typing import Any, Dict, List
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    def __init__(
        self,
        conv_dir: str,
        shuffle: bool = False
    ):
        self.conv_dir = pathlib.Path(conv_dir)

        self.conversations: List[Dict[str, Any]] = []
        for conv_path in self.conv_dir.iterdir():
            if conv_path.suffix == '.json':
                conversation: Dict[str, Any] = json.loads(
                    conv_path.read_bytes()
                )
                conversation['conv'] = f"User: {conversation.pop('tweet')}\n\n{conversation['conv']}"

                self.conversations.append(conversation)

        # Shuffle the conversations
        if shuffle: random.shuffle(self.conversations)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index) -> Dict[str, Any]:
        return self.conversations[index]
