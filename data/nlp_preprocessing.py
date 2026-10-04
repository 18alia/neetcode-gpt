import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        positive= [sentence.split() for sentence in positive]
        #print(positive)
        negative= [sentence.split() for sentence in negative]
        combined= [word for sentence in positive for word in sentence] + [word for sentence in negative for word in sentence]
        #print(combined)
        words= list(dict.fromkeys(combined))
        sorted_words= sorted(words)
        print(sorted_words)
        combined_tensors= []
        for sentence in positive:
            embed= torch.zeros(len(sentence))
            for i,word in enumerate(sentence):
                embed[i]= float(sorted_words.index(word)+1.0)
            combined_tensors.append(embed)
        for sentence in negative:
            embed= torch.zeros(len(sentence))
            for i,word in enumerate(sentence):
                embed[i]= float(sorted_words.index(word)+1.0)
            combined_tensors.append(embed)
        output=  nn.utils.rnn.pad_sequence(combined_tensors,padding_value=0,
                batch_first=True)
        return output
        
